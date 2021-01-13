import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import pysal as ps
import shapefile
import contextily as ctx

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

from tespy.components import source, sink, heat_exchanger_simple, pipe
from tespy.connections import connection, bus, ref
from tespy.networks import network

from sub_consumer import (lin_consum_closed as lc,
                          lin_consum_open as lo,
                          fork as fo)




from bokeh.layouts import row, grid
from bokeh.models import CustomJS, ColumnDataSource, HoverTool, LogColorMapper
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models.tools import *
from bokeh.plotting import *
from bokeh.palettes import RdYlBu11 as palette
import bokeh.io
from bokeh.io import output_notebook, show
import webbrowser


import itertools
from scipy.spatial import cKDTree
from operator import itemgetter
import shapely.geometry as geom
from shapely.geometry import Point, LineString


import datetime as dt
from datetime import datetime, date

import sys
sys.setrecursionlimit(3000)




def Get_age_ID_pipe(df_pipes):
    return(df_pipes
         .assign(ID = df_pipes.index+1)
         .assign(Age = lambda x: date.today().year - pd.to_datetime(x["BUILD_DATE"]).dt.year)
         .assign(BUILD_DATE = lambda x: pd.to_datetime(x["BUILD_DATE"]).dt.year))

def Get_ID(df_users):
    return(df_users
         .assign(USER_ID = df_users.index+1))
         
def ckdnearest(gdfA, gdfB, gdfB_cols=['ID']):
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdfA.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

def Get_coords(df_pipes,df_users):
        return(df_pipes
        
        .assign(node_ups = gpd.GeoSeries([Point(list(pt['geometry'].coords)[0]) for i,pt in df_pipes.iterrows()]))
        .assign(node_dws = gpd.GeoSeries([Point(list(pt['geometry'].coords)[-1]) for i,pt in df_pipes.iterrows()]))
        .merge(df_users[["ID","USER_ID"]] , left_on='ID', right_on='ID',how="outer")
    )    
 
def getLineCoords(row, geom, coord_type):
    """Returns a list of coordinates ('x' or 'y') of a LineString geometry"""
    if coord_type == 'x':
        return list( row[geom].coords.xy[0] )
    elif coord_type == 'y':
        return list( row[geom].coords.xy[1] )

def DHS_map(pipes,users,start_pipe_ID,linewidth_factor = 1/30, markersize_users_factor= 1/5000, markersize_plant = 500, color_presentation = "Age", linewidth_column = "DIMENSION",cons_annotation_column = "USER_ID",cons_annotation=False):
    df_start_pipe = pipes.loc[pipes["ID"]== start_pipe_ID]
    pipes_g = gpd.GeoDataFrame(pipes, geometry= pipes["geometry"])
    users_g = gpd.GeoDataFrame(users, geometry= users["geometry"])
    plant = df_start_pipe[["node_ups"]]
    plant = gpd.GeoDataFrame(plant, geometry="node_ups")
    fig, ax = plt.subplots(figsize=(15,15))
    pipes_g.plot(column=color_presentation,cmap='RdYlGn_r',
            ax=ax, 
            legend=True,
            legend_kwds={'label': color_presentation,'orientation': "vertical"},
            linewidth=pipes[linewidth_column]*linewidth_factor)
    ax.set_axis_off()
    users_g.plot(color="blue",
            ax=ax, 
            markersize=users["Power"]*markersize_users_factor)
    if cons_annotation == True:
        users.apply(lambda x: ax.annotate(text=round(x[cons_annotation_column],1), xy=x.geometry.centroid.coords[0], xytext=(10,10), textcoords = "offset points",arrowprops=dict(arrowstyle="->")),axis=1)
    else:
        pass

    plant.plot(marker='*',ax=ax, color='black', markersize=markersize_plant)
    plant.apply(lambda x: ax.annotate(text="Plant", xy=x.node_ups.centroid.coords[0], xytext=(50,0), textcoords = "offset points", arrowprops=dict(arrowstyle="<->") ),axis=1); 

    ctx.add_basemap(ax,url=ctx.sources.OSM_C, zoom=16)
    plt.show()



def selection_tool(df_pipes,df_users):
    # Create the color mapper
    color_mapper = LogColorMapper(palette=palette)

    
    df_pipes['x'] = df_pipes.apply(getLineCoords, geom='geometry', coord_type='x', axis=1)
    df_pipes['y'] = df_pipes.apply(getLineCoords, geom='geometry', coord_type='y', axis=1)
    df_nodes=df_pipes[["ID","node_ups"]].copy()
    df_nodes['x'] = df_pipes.apply(getLineCoords, geom='node_ups', coord_type='x', axis=1)
    df_nodes['y'] = df_pipes.apply(getLineCoords, geom='node_ups', coord_type='y', axis=1)

    df_users['x'] = df_users.apply(getLineCoords, geom='geometry', coord_type='x', axis=1)
    df_users['y'] = df_users.apply(getLineCoords, geom='geometry', coord_type='y', axis=1)

    #pipes source
    m_df = df_pipes[["ID","x","y","BUILD_DATE","DIMENSION"]].copy()
    msource=ColumnDataSource(m_df)

    #defining x,y and ID for upstream nodes and source
    n_df = df_nodes[['ID',"x","y"]].copy()
    
    def node_coords(n_df):
        n_df['x'] = n_df['x'].str.get(0)
        n_df['y'] = n_df['y'].str.get(0)
        #n_df['ID'] = n_df['ID'].str.get(0)
        n_df['x'] = n_df['x'].astype(float)
        n_df['y'] = n_df['y'].astype(float)
        n_df['x'] = n_df['x'].apply(lambda x: '{:.2f}'.format(x))
        n_df['y'] = n_df['y'].apply(lambda x: '{:.2f}'.format(x))
        return(n_df)

    n_df = node_coords(n_df)

    x=list(n_df['x'])
    y=list(n_df['y'])
    ID=list(n_df['ID'])
    s1 = ColumnDataSource(data=dict(x=x, y=y))

    #uporabniki
    u_df = df_users[["x","y"]].copy()
    u_df['x'] = u_df['x'].str.get(0)
    u_df['y'] = u_df['y'].str.get(0)
    usource = ColumnDataSource(u_df)



    #figure
    p1 = figure(plot_width=1000, plot_height=1200
                , tools=["lasso_select",'wheel_zoom','box_zoom'], title="District heating")




    #ploting upstream nodes
    p1.circle('x', 'y', source=usource, alpha=0.6,size=10)
    p1.circle('x', 'y', source=s1, alpha=0.6)



    #second figure
    s2 = ColumnDataSource(data=dict(x=[], y=[],))
    p2 = figure(plot_width=1000, plot_height=1200, x_range=(0, 1), y_range=(0, 1),tools="", title="Selected pipes")
    p2.circle('x', 'y', source=s2, alpha=0.6)

    columns = [TableColumn(field ="x",  title = "X axis"),
               TableColumn(field ="y",  title = "Y axis"),
               ]

    table = DataTable(source =s2, columns = columns, width =300, height = 500)


    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, table=table), code="""
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = s2.data;
            d2['x'] = []
            d2['y'] = []
            var text = "";
            for (var i = 0; i < inds.length; i++) {
                d2['x'].push(d1['x'][inds[i]])
                d2['y'].push(d1['y'][inds[i]])
                text = text  + d1['x'][inds[i]] + "," + d1['y'][inds[i]] + '\\n'

            }


            var filename = "SelectedPipes.txt";

            var blob = new Blob([text], {type:'text/plain'});

            var elementExists = document.getElementById("link");

            if(elementExists){
                elementExists.href = window.URL.createObjectURL(blob);
            }
            else{
                var link = document.createElement("a");
                link.setAttribute("id", "link");
                link.download = filename;
                link.innerHTML = "Select Pipes";
                link.href = window.URL.createObjectURL(blob);
                document.body.appendChild(link);
            }




            s2.change.
            emit();
            table.change.emit();
        """)
    )


    p1.multi_line('x', 'y', source=msource, color='gray', line_width=3)
    
    my_hover = HoverTool()
    my_hover.tooltips = [('ID', '@ID'), ('Diameter', '@DIMENSION'), ('Year', '@BUILD_DATE'),('Next', '@PipeNext'),('Previous', '@Previous')]
    p1.add_tools(my_hover)



    layout = row(p1, table)

    show(layout, browser=None, new='tab', notebook_handle=False)


 


def get_selected_pipes(df_pipes,path,name_downloaded):
    SelectedPipes = pd.read_csv(path + name_downloaded +".txt", header = None)
    SelectedPipes.columns = ["x", "y"]

    def find_selected_pipes(df_pipes):
        df_pipes['node_ups_x'] = df_pipes.apply(getLineCoords, geom='node_ups', coord_type='x', axis=1)
        df_pipes['node_ups_y'] = df_pipes.apply(getLineCoords, geom='node_ups', coord_type='y', axis=1)

        df_pipes['node_ups_x'] = df_pipes['node_ups_x'].str.get(0)
        df_pipes['node_ups_y'] = df_pipes['node_ups_y'].str.get(0)

        df_pipes['node_ups_y'] = df_pipes['node_ups_y'].apply(pd.to_numeric, errors='coerce')
        df_pipes['node_ups_x'] = df_pipes['node_ups_x'].apply(pd.to_numeric, errors='coerce')

        df_pipes['node_ups_y'] = df_pipes['node_ups_y'].apply(lambda x: '{:.2f}'.format(x))
        df_pipes['node_ups_x'] = df_pipes['node_ups_x'].apply(lambda x: '{:.2f}'.format(x))

        df_pipes['node_ups_y'] = df_pipes['node_ups_y'].apply(pd.to_numeric, errors='coerce')
        df_pipes['node_ups_x'] = df_pipes['node_ups_x'].apply(pd.to_numeric, errors='coerce')

        return df_pipes

    df_pipes = find_selected_pipes(df_pipes)

    selected_pipes = pd.merge(SelectedPipes, df_pipes,  how='left', left_on=['y'], right_on = ['node_ups_y'])

    selected_pipes = selected_pipes[selected_pipes['ID'].notna()]
    return(selected_pipes)



def delete_pipes(delete_pipes_IDs,selected_pipes):
    for i in delete_pipes_IDs:
        selected_pipes = selected_pipes[selected_pipes.ID != i]
    return selected_pipes

def delete_df_pipes_columns(df_pipes):
    df_pipes = df_pipes.drop(columns = ["x","y","node_ups_x","node_ups_y"])
    return(df_pipes)

def calculation_pipes(selected_pipes,df_pipes,df_users):
    df_pipes_c=selected_pipes
    df_pipes_c = df_pipes_c.drop(columns=["x_x","y_x","node_ups_x","node_ups_y","x_y","y_y"])

    df_pipes_c_user=df_pipes_c["USER_ID"]
    df_users_c=pd.merge(df_pipes_c_user, df_users, left_on='USER_ID', right_on='USER_ID',how="inner")
    df_users_c = df_users_c.drop_duplicates(subset=['USER_ID'])
    df_users_c = df_users_c.drop(columns=['x','y'])



    df_pipes_c['ID'] = df_pipes_c['ID'].astype(int)   

    return(df_pipes_c,df_users_c)

def pipe_data(df_pipe_data):
    df_pipe_data["DIMENSION"]=df_pipe_data["DIMENSION"].astype(str)
    df_pipe_data["Roughness"]=df_pipe_data["Roughness"].astype(float)
    df_pipe_data["Ch"]=df_pipe_data["Ch"].astype(float)
    df_pipe_data["type"]=df_pipe_data["DIMENSION"]+df_pipe_data["Type"]
    df_pipe_data=df_pipe_data[["type","Roughness","Ch"]]
    return df_pipe_data

def pipes_characteristics(df_pipe_data,Roughness,df_pipes_c):
    df_pipes_data=df_pipes_c.copy()
    df_pipes_data["DIMENSION"]=df_pipes_data["DIMENSION"].astype(int)
    df_pipes_data["DIMENSION"]=df_pipes_data["DIMENSION"].astype(str)
    df_pipes_data["type"]=df_pipes_data["DIMENSION"]+df_pipes_data["POSITION"]

    df_pipes_data=df_pipes_data[["BUILD_DATE","DIMENSION","ID","LENGTH","POSITION","Age"]]

    df_pipes_data["DIMENSION"]=df_pipes_data["DIMENSION"].astype(str)
    df_pipes_data["type"]=df_pipes_data["DIMENSION"]+df_pipes_data["POSITION"]

    

    df_pipes_data=pd.merge(df_pipes_data, df_pipe_data, left_on='type', right_on='type',how="outer")

    df_pipes_data = df_pipes_data[df_pipes_data['ID'].notna()]

    
    df_pipes_data["Roughness"]=df_pipes_data["Roughness"]*(1.05)**df_pipes_data["Age"]
    df_pipes_data["Ch"]=df_pipes_data["Ch"]*(1.01)**df_pipes_data["Age"]
    df_pipes_data["Ch"]=(df_pipes_data["Ch"]+0.3)*15
    df_pipes_data["ID"]=df_pipes_data["ID"].astype(int)
    Diameter = {"20":0.0217, "25":0.0273, "32":0.036, "40":0.0419, "50":0.0539, "65":0.0697,  "80":0.0825, "100":0.1071, "125":0.1325, "150":0.1603, "200":0.2101, "250":0.263}


    df_pipes_data["Diameter"]=df_pipes_data["DIMENSION"].map(Diameter)
    
    Roughness[0] = Roughness[0].astype(str)

    Roughness = Roughness.set_index([0])
    Roughness = Roughness.to_dict()[1]
    df_pipes_data["BUILD_DATE"] = df_pipes_data["BUILD_DATE"].astype(int)
    df_pipes_data["BUILD_DATE"] = df_pipes_data["BUILD_DATE"].astype(str)
    df_pipes_data["Roughness"]=df_pipes_data["BUILD_DATE"].map(Roughness)
    df_pipes_data["BUILD_DATE"] = df_pipes_data["BUILD_DATE"].astype(int)
    df_pipes_data["Roughness"] = df_pipes_data["Roughness"]/1000
    df_pipes_data=df_pipes_data.drop_duplicates(subset="ID")
    return(df_pipes_data)

def connections_ready_for_checking_downstream(df_pipes_c):
    df_connections = df_pipes_c.copy()
    df_connections=df_connections[["ID","node_ups","node_dws"]].copy()
    df_connections['pipePrevious'] = 'undetermined'
    df_connections['pipeNext'] = 'undetermined'

    df_connections=df_connections.set_index('ID')
    df_connections['ID'] = df_connections.index
    df_connections['node_ups'] = df_connections['node_ups'].astype(str)
    df_connections['node_dws'] = df_connections['node_dws'].astype(str)
    df_connections=df_connections.drop_duplicates(subset="ID",keep="first")
    return df_connections

def check_downstream(pipe_id, pipes):
    """Set next and previous pipe ID for all downstream pipes"""

    endpoint = pipes.loc[pipe_id, 'node_dws']
    next_ids = list(pipes.loc[pipes['node_ups'] == endpoint]['ID'])
    reversed_ids = list(pipes[(pipes['node_dws'] == endpoint) & (pipes['ID'] != pipe_id)]['ID'])
    
    for rev_id in reversed_ids:
        pipes.loc[rev_id, 'node_ups'], pipes.loc[rev_id, 'node_dws'] = pipes.loc[rev_id, 'node_dws'], pipes.loc[rev_id, 'node_ups']
    next_ids = next_ids + reversed_ids

    pipes.at[pipe_id, 'pipeNext'] = next_ids
   
    for nxt_id in next_ids:
        pipes.loc[nxt_id, 'pipePrevious'] = pipe_id
        check_downstream(nxt_id, pipes)

def get_forks_data(df_connections):
    
    df_connections["fork"]=df_connections["pipeNext"].astype(str)
    df_connections["fork"]=df_connections["fork"].str.contains(",", regex=False)
    len_df_forks=3*len(df_connections["fork"]==True)
    df_connections_saved = df_connections.copy()
    df_connections=df_connections.drop(columns=["node_ups","node_dws","ID"])


    df_connections=df_connections.reset_index()

    df_forks_data = pd.DataFrame()
    df_forks_data = df_connections.loc[(df_connections['fork'] == True), ['ID','pipeNext','pipePrevious']]
    return (df_forks_data,df_connections_saved)

def get_fork_exit(df_connections,df_users_c,df_forks_data):

    new_rows = []
    for index, row in df_forks_data.iterrows():
        lines = [row["pipePrevious"], row['ID'], "K"+str(row["ID"])]
        new_rows.append(lines)
        lines1 = [row["ID"], "K"+str(row["ID"]), row["pipeNext"][0] ]
        new_rows.append(lines1)
        lines2 = [row["ID"], "K"+str(row["ID"]), row["pipeNext"][1] ]
        new_rows.append(lines2)

    df_forks= pd.DataFrame(new_rows, columns= ["ID_previous", "ID", "ID_next" ])

    df_connections = df_connections.drop(df_connections[df_connections["fork"] == True ].index)
    df_connections=df_connections.explode('pipeNext')
    df_connections=df_connections.rename(columns={"pipePrevious":"ID_previous", "pipeNext":"ID_next"})
    df_connections = df_connections.append(df_forks)
    df_connections=df_connections.reset_index()

    df_users_c=df_users_c.drop_duplicates(subset="geometry")

    df_users_c_pure=df_users_c[["ID","USER_ID","Power"]]
    df_users_c_pure["USER_ID"]=df_users_c_pure["USER_ID"].astype(int)

    df_connections=pd.merge(df_connections, df_users_c_pure, left_on='ID', right_on='ID',how="outer")

    df_connections['dup_number'] = df_connections.groupby(['ID']).cumcount()+1

    def f(row):
        if row['dup_number'] == 1:
            val = 1
        elif row['dup_number'] == 2:
            val = 2
        else:
            val = 1
        return val

    df_connections["ForkExit"] = df_connections.apply(f, axis=1)

    df_connections.drop(['dup_number'], axis=1, inplace=True)
    return(df_connections)

def get_closest_pipes(df_connections,df_pipes_c,df_users_c,df_users):
    df_connections = df_connections.drop(columns = ["node_dws","node_ups"])
    df_connections_user=pd.merge(df_connections, df_pipes_c, left_on='ID', right_on='ID',how="inner")

    df_connections_user=df_connections_user[df_connections_user["ID_next"].isnull()]
    df_connections_user=df_connections_user.drop_duplicates(subset="ID")
    df_connections_user=df_connections_user[["ID","node_dws"]]

    df_users_c=df_users_c.drop_duplicates(subset="geometry")

    df_users_c_pure=df_users_c[["ID","USER_ID","Power"]]
    df_users_c_pure["USER_ID"]=df_users_c_pure["USER_ID"].astype(int)

    df_users_c_pure["USER_ID"]=df_users_c_pure["USER_ID"].astype(int)

    df_users_user=pd.merge(df_users_c_pure, df_users, left_on='USER_ID', right_on='USER_ID',how="inner")

    df_users_user=df_users_user[["USER_ID","geometry","Power_x"]]

    df_users_user=df_users_user.rename(columns={"Power_x":"Power"})

    #df_users_user=df_users_user.rename(columns={"ID_x":"ID"})

    #attaching consumers to closest end pipes based on the closest geometries
    gpd2 = gpd.GeoDataFrame(df_connections_user, geometry= df_connections_user["node_dws"])
    gpd1= gpd.GeoDataFrame(df_users_user, geometry= df_users_user["geometry"])
    gpd1["USER_ID"]=gpd1["USER_ID"].astype(str)
    gpd2["ID"]=gpd2["ID"].astype(str)


    gpd2=gpd2.reset_index()
    gpd1=gpd1.reset_index()

    def ckdnearest(gdA, gdB):
        nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        gdf = pd.concat(
            [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
             pd.Series(dist, name='dist')], axis=1)
        return gdf

    closest_pipes=ckdnearest(gpd1, gpd2)


    closest_pipes=closest_pipes.rename(columns={"Power_x":"Power"})
    return closest_pipes

def end_pipes_f(df_connections,closest_pipes,user_p_drop):
    end_pipes=df_connections.loc[df_connections["ID_next"].isnull()]
    end_pipes=df_connections.loc[df_connections["ID_next"].isnull()]
    end_pipes["Power"]= end_pipes.loc[end_pipes["Power"]== np.nan]
    users=closest_pipes.groupby(["ID"]).sum()
    users=users.reset_index()
    users["ID"]=users["ID"].astype(int)
    users["USER_ID"]=users["ID"]+55555
    end_pipes=pd.merge(end_pipes, users, left_on='ID', right_on='ID',how="outer")
    end_pipes=end_pipes.drop_duplicates(subset="ID")
    end_pipes=end_pipes[["ID","ID_previous","ID_next","Power_y","USER_ID_y"]]
    end_pipes=end_pipes.rename(columns={"Power_y":"Power","USER_ID_y":"USER_ID" })
    end_pipes["Power"].fillna(0, inplace=True)
    end_pipes["Power"]=end_pipes["Power"]
    end_pipes["USER_ID"]=end_pipes["ID"]+55555
    end_pipes["USER_ID"]=end_pipes["USER_ID"]-55555
    end_pipes["T_min"]=70
    end_pipes["p_drop"]=user_p_drop
    return end_pipes

def get_pipe_user_connections(df_connections):
    df_pipe_user=df_connections[df_connections['USER_ID'].notna()]
    return(df_pipe_user)

def get_pipe_pipe_connections(df_connections):
    df_pipe_pipe = df_connections.copy()
    df_pipe_pipe["ID"]=pd.to_numeric(df_pipe_pipe["ID"], errors='coerce')
    df_pipe_pipe["ID_next"]=pd.to_numeric(df_pipe_pipe["ID_next"], errors='coerce')
    df_pipe_pipe = df_pipe_pipe[df_pipe_pipe['ID_next'].notna()]
    df_pipe_pipe = df_pipe_pipe[df_pipe_pipe['ID'].notna()]
    return(df_pipe_pipe)

def get_fork_pipe_connections(df_connections):
    df_fork_pipe= df_connections.copy()
    df_fork_pipe = df_fork_pipe[df_fork_pipe['ID'].apply(lambda x: type(x) == str)]
    return(df_fork_pipe)

def get_pipe_fork_connections(df_connections):
    df_pipe_fork=df_connections.copy()
    df_pipe_fork = df_pipe_fork[df_pipe_fork['ID_next'].apply(lambda x: type(x) == str)]
    return(df_pipe_fork)
 
#defining pipe feed
def pipe_feed_definition(df_pipes_data):
    pf = {}

    for index, row in df_pipes_data.iterrows(): 
        #pipe_name = "pipe" + str(row["ID"]) + "_feed"
        ks = row["Roughness"]
        L = row["LENGTH"]
        D = row["Diameter"]
        kA = row["Ch"]

        pf[row['ID']] = pipe("pipe" + str(row["ID"]) + "_feed", ks = ks, L = L, D = D, kA = kA)
    return pf


#defining return pipes   
def pipe_return_definition(df_pipes_data):
    pb = {}    

    for index, row in df_pipes_data.iterrows(): 
        #pipe_name_back = "pipe" + str(row["ID"]) + "_back"
        ks = row["Roughness"]
        L = row["LENGTH"]
        D = row["Diameter"]
        kA = row["Ch"]

        pb[row['ID']] = pipe("pipe" + str(row["ID"]) + "_back", ks = ks, L = L, D = D, kA = kA)
    return pb



#defining forks
def forks(df_forks_data):
    k = {}

    for index, row in df_forks_data.iterrows(): 
        fork_name = "K" + str(row["ID"])


        k[row['ID']] = fo(fork_name, 2)
        k[row['ID']].char_warnings=False
    return k

def add_subsys(k,nw):
    nw.add_subsys(*[x[1] for x in k.items()])
    return nw

def consumers(end_pipes):
    cons = {}


    for index, row in end_pipes.iterrows(): 
        cons_name = "consumer" + str(int(row["USER_ID"]))

        cons[int(row['USER_ID'])] = heat_exchanger_simple(cons_name)
        cons[int(row["USER_ID"])].set_attr(Q=-row["Power"])
    return cons


def define_connections_so_start(start_pipe_ID,pf,so):
    start=pf[start_pipe_ID]
    so_start = connection(so, 'out1', start, 'in1',  fluid={'water': 1})
    return so_start


def define_connections_start_si(start_pipe_ID,pb,si):        
    end=pb[start_pipe_ID] 
    start_si = connection(end, 'out1', si, 'in1')
    return start_si


def define_connections_pipe_pipe_feed(df_pipe_pipe,pf):
    pipe_pipe_f = {}  
    df_pipe_pipe["ID"]=df_pipe_pipe["ID"].astype(int)
    for index, row in df_pipe_pipe.iterrows(): 
        pipe_pipe_f[row["ID"]]=connection(pf[row["ID"]], "out1", pf[row["ID_next"]], "in1")    
    return pipe_pipe_f


def define_connections_pipe_pipe_back(df_pipe_pipe,pb):
    pipe_pipe_b = {}  
    df_pipe_pipe["ID_next"]=df_pipe_pipe["ID_next"].astype(int)
    for index, row in df_pipe_pipe.iterrows(): 
        pipe_pipe_b[row["ID_next"]]=connection(pb[row["ID_next"]], "out1", pb[row["ID"]], "in1")
    return pipe_pipe_b


def define_connections_pipe_user(df_pipe_user,end_pipes,pf,cons):
    pipe_user = {}  
    df_pipe_user["USER_ID"]=df_pipe_user["USER_ID"].astype(int)
    for index, row in end_pipes.iterrows():    
        pipe_user[int(row["ID"])]=connection(pf[row["ID"]], "out1", cons[row["USER_ID"]], "in1", design=["T"])
    return pipe_user


def define_connections_user_pipe(end_pipes,pb,cons,pipe_user):    
    user_pipe = {}     
    for index, row in end_pipes.iterrows(): 
        user_pipe[int(row["USER_ID"])]=connection(cons[row["USER_ID"]], "out1", pb[row["ID"]], "in1", p=ref(pipe_user[row["ID"]], 1, row["p_drop"]))
    return user_pipe


def define_connections_fork_pipe_1_feed(df_fork_pipe,k,pf):
    df_fork_pipe["Valve"]=df_fork_pipe["ForkExit"]-1
    df_fork_pipe1=df_fork_pipe.loc[df_fork_pipe["ForkExit"]==1]
    fork_pipe_f1 = {}  
    for index, row in df_fork_pipe1.iterrows(): 
        fork_pipe_f1[row["ID_next"]]=connection(k[row["ID_previous"]].comps["splitter"], "out"+str(row["ForkExit"]), pf[row["ID_next"]], "in1")
    return fork_pipe_f1


def define_connections_fork_pipe_1_back(df_fork_pipe,k,pb):
    df_fork_pipe["Valve"]=df_fork_pipe["ForkExit"]-1
    df_fork_pipe1=df_fork_pipe.loc[df_fork_pipe["ForkExit"]==1]
    fork_pipe_b1 = {}  
    for index, row in df_fork_pipe1.iterrows(): 
        fork_pipe_b1[row["ID_next"]]=connection(pb[row["ID_next"]], "out1", k[row["ID_previous"]].comps["valve_"+str(row["Valve"])], "in1")
    return fork_pipe_b1


def define_connections_fork_pipe_2_feed(df_fork_pipe,k,pf):
    df_fork_pipe["Valve"]=df_fork_pipe["ForkExit"]-1
    df_fork_pipe2=df_fork_pipe.loc[df_fork_pipe["ForkExit"]==2]    
    fork_pipe_f2 = {}      
    for index, row in df_fork_pipe2.iterrows(): 
        fork_pipe_f2[row["ID_next"]]=connection(k[row["ID_previous"]].comps["splitter"], "out"+str(row["ForkExit"]), pf[row["ID_next"]], "in1")   
    return fork_pipe_f2


def define_connections_fork_pipe_2_back(df_fork_pipe,k,pb):
    df_fork_pipe["Valve"]=df_fork_pipe["ForkExit"]-1
    df_fork_pipe2=df_fork_pipe.loc[df_fork_pipe["ForkExit"]==2]    
    fork_pipe_b2 = {}  
    for index, row in df_fork_pipe2.iterrows(): 
        fork_pipe_b2[row["ID_next"]]=connection(pb[row["ID_next"]], "out1", k[row["ID_previous"]].comps["valve_"+str(row["Valve"])], "in1")
    return fork_pipe_b2


def define_connections_pipe_fork_feed(df_pipe_fork,df_fork_pipe,k,pf):
    df_pipe_fork_nextnext=pd.merge(df_pipe_fork, df_fork_pipe, left_on='ID', right_on='ID_previous',how="outer")
    df_pipe_fork_nextnext=df_pipe_fork_nextnext[["ID_x","ID_previous_x","ID_next_x","ID_next_y"]]
    df_pipe_fork_nextnext=df_pipe_fork_nextnext.rename(columns={"ID_x":"ID","ID_previous_x":"ID_previous", "ID_next_x":"ID_next", "ID_next_y":"ID_next_next"})
    df_pipe_fork_nextnext=df_pipe_fork_nextnext.iloc[::2,:]
    pipe_fork_f = {}  
    for index, row in df_pipe_fork_nextnext.iterrows():     
        pipe_fork_f[row["ID"]]=connection(pf[row["ID"]], "out1", k[row["ID"]].comps["splitter"], "in1")
    return pipe_fork_f

def define_connections_pipe_fork_back(df_pipe_fork,df_fork_pipe,k,pb,fork_pipe_b1):
    df_pipe_fork_nextnext=pd.merge(df_pipe_fork, df_fork_pipe, left_on='ID', right_on='ID_previous',how="outer")
    df_pipe_fork_nextnext=df_pipe_fork_nextnext[["ID_x","ID_previous_x","ID_next_x","ID_next_y"]]
    df_pipe_fork_nextnext=df_pipe_fork_nextnext.rename(columns={"ID_x":"ID","ID_previous_x":"ID_previous", "ID_next_x":"ID_next", "ID_next_y":"ID_next_next"})
    df_pipe_fork_nextnext=df_pipe_fork_nextnext.iloc[::2,:]
    pipe_fork_b = {}
    for index, row in df_pipe_fork_nextnext.iterrows(): 
        pipe_fork_b[row["ID"]]=connection(k[row["ID"]].comps["merge"], "out1", pb[row["ID"]], "in1", p=ref(fork_pipe_b1[row["ID_next_next"]], 1, -5e4))
    return pipe_fork_b

def add_connections(nw,so_start,start_si,pipe_pipe_f,pipe_pipe_b,pipe_user,user_pipe,fork_pipe_f1,fork_pipe_b1,fork_pipe_f2,fork_pipe_b2,pipe_fork_f,pipe_fork_b):
    nw.add_conns(so_start,start_si)
    nw.add_conns(*[x[1] for x in pipe_pipe_f.items()])
    nw.add_conns(*[x[1] for x in pipe_pipe_b.items()])
    nw.add_conns(*[x[1] for x in pipe_user.items()])
    nw.add_conns(*[x[1] for x in user_pipe.items()])
    nw.add_conns(*[x[1] for x in fork_pipe_f1.items()])
    nw.add_conns(*[x[1] for x in fork_pipe_b1.items()])
    nw.add_conns(*[x[1] for x in fork_pipe_f2.items()])
    nw.add_conns(*[x[1] for x in fork_pipe_b2.items()])
    nw.add_conns(*[x[1] for x in pipe_fork_f.items()])
    nw.add_conns(*[x[1] for x in pipe_fork_b.items()])
    return nw

def get_results(df_pipes_c,end_pipes,df_users_c,pf,user_pipe,pipe_user,df_users):
    def pipe_heat_losses(row):
        return pf[row["ID"]].Q.val
    df_pipes_results = df_pipes_c.copy()
    df_pipes_results["pipe_heat_losses"]=df_pipes_results.apply(pipe_heat_losses, axis=1)
    df_pipes_results["pipe_heat_losses"] = df_pipes_results["pipe_heat_losses"]*(-1)

    df_pipes_results = df_pipes_results.drop_duplicates(subset=["ID"])
    def get_p_return(row):
        return user_pipe[row["ID"]].p.val
    def get_T_feed(row):
        return pipe_user[row["ID"]].T.val
    end_pipes_results = end_pipes[["ID","Power"]]
    end_pipes_results["p_return"]= end_pipes.apply(get_p_return, axis=1)
    end_pipes_results["T_feed"]= end_pipes.apply(get_T_feed, axis=1)
    df_users_results=pd.merge(df_users[["ADRESS","geometry","ID","USER_ID"]], end_pipes_results, left_on='ID', right_on='ID',how="inner")
    df_users_results = gpd.GeoDataFrame(df_users_results, geometry="geometry")
    return(
        df_pipes_results,
        df_users_results)


def print_results(heat_consumer,heat_losses,so_start):
    return(
    print("PRINTED RESULTS:\n"
          'Heat demand consumer:', round(heat_consumer.P.val/1000,2), "kW","\n", 
          'Network losses:', round(heat_losses.P.val/1000,2), "kW\n",
          'Network losses:',round(heat_losses.P.val/(heat_consumer.P.val+heat_losses.P.val)*100,2), "%\n",
          "Source feed temperature:", round(so_start.T.val,2), "°C\n",
          "Source feed mass flow:", round(so_start.m.val,2), "kg/s\n",
          "Source feed pressure:", round(so_start.p.val,2), "bar"
    ))