import math
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import imageio
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Calculates the distance between two points on a sphere
    
    Parameters
        lat1: float latitude of the first point
        lon1: float longitude of the first point
        lat2: float latitude of the second point
        lon2: float longitude of the second point
    
    Returns
        distance: float distance between the two points
    """
    R = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance/1.609

def calc_distance(Cities, average_mph:int = 60) -> list:
    """
    Calculates the distance between each city in the list of cities
    
    Parameters
        Cities: list of tuples of the form (latitude, longitude)
        average_mph: int average speed of the vehicle in miles per hour
    
    Returns
        distances: list of tuples of the form (city1, city2, distance)
    """
    distances = []

    for i, city_i in enumerate(Cities):
        for j, city_j in enumerate(Cities, start=i+1):
            distances.append((i, j, haversine(*city_i , *city_j)/average_mph))

    for i, j, distance in distances:
        if distance == 0:
            #print(i, j, distance)
            distances.remove((i, j, distance))
    
    return distances

def plot_cities(Cities: list, save: bool = True) -> None:
    """
    Plots the cities on a map
    
    Parameters
        Cities: list of tuples of the form (latitude, longitude)
        
    Returns
        None
    """
    df = pd.DataFrame(Cities)
    geometry = [Point(xy) for xy in zip(df[1], df[0])]

    map = gpd.read_file(r'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\data\ne_110m_admin_1_states_provinces.shp')

    geo_df = gpd.GeoDataFrame(
        df,
        geometry = geometry
        )
    
    fig, ax = plt.subplots(figsize=(10,5))
    map.plot(ax=ax, alpha=0.4,color='grey')
    geo_df.plot(ax=ax)
    plt.xlim(-130,-65)
    plt.ylim(23,50)
    plt.title('Cities')
    plt.tight_layout()  
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    if save == True:
        plt.savefig(fr'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\images\cities.png', 
                transparent = False,  
                facecolor = 'white'
               )

    plt.show()


def plot_route(cities, best_state) -> None:
    """
    Plots the cities on a map
    
    Parameters
        Cities: list of tuples of the form (latitude, longitude)
        best_state: list of the best route
        
    Returns
        None
    """
    df = pd.DataFrame(cities)
    geometry = [Point(xy) for xy in zip(df[1], df[0])]

    map = gpd.read_file(r'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\data\ne_110m_admin_1_states_provinces.shp')

    geo_df = gpd.GeoDataFrame(df,
        geometry = geometry)

    fig, ax = plt.subplots(figsize=(10,5))
    map.plot(ax=ax, alpha=0.4,color='grey')

    for i, line in enumerate(best_state):
        if i < 47:
            x0, y0 = cities[best_state[i+1]]
            x1, y1 = cities[best_state[i]]
            ax.plot([y0, y1], [x0, x1], color='red', linewidth=2, zorder=3)

    x0, y0 = cities[best_state[-1]]
    x1, y1 = cities[best_state[0]]
    ax.plot([y0, y1], [x0, x1], color='red', linewidth=2, zorder=3)    

    geo_df.plot(ax=ax)

    plt.xlim(-130,-65)
    plt.ylim(23,50)
    plt.show()

def create_frame(iter, cities, distances):
    RANDOM_STATE = 42

    fitness_coords = mlrose.TravellingSales(coords = cities)
    fitness_dists = mlrose.TravellingSales(distances = distances)

    problem_fit = mlrose.TSPOpt(length = 48, fitness_fn = fitness_coords, maximize=False)

    print(f'Running... iter: {iter}')
    best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_iters=iter, max_attempts=1000, restarts=100, random_state = RANDOM_STATE)

    df = pd.DataFrame(cities)
    geometry = [Point(xy) for xy in zip(df[1], df[0])]

    map = gpd.read_file(r'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\data\ne_110m_admin_1_states_provinces.shp')

    geo_df = gpd.GeoDataFrame(df,
        geometry = geometry)

    fig, ax = plt.subplots(figsize=(10,5))
    map.plot(ax=ax, alpha=0.4,color='grey')

    for i, line in enumerate(best_state):
        if i < 47:
            x0, y0 = cities[best_state[i+1]]
            x1, y1 = cities[best_state[i]]
            ax.plot([y0, y1], [x0, x1], color='red', linewidth=2, zorder=3)

    x0, y0 = cities[best_state[-1]]
    x1, y1 = cities[best_state[0]]
    ax.plot([y0, y1], [x0, x1], color='red', linewidth=2, zorder=3)    

    geo_df.plot(ax=ax)

    plt.xlim(-130,-65)
    plt.ylim(23,50)

    plt.title(f'Random Hill Climb - {iter} iterations')
    plt.tight_layout()
    plt.text(-129, 24, f'Fitness: {round(best_fitness, 2)}', fontsize=12)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.savefig(fr'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\images\img_{iter}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()


def create_gif(iters, cities) -> None:
    frames = []

    for i, iter in enumerate(iters):
        create_frame(iter)

    frames = []
    for i, iter in enumerate(iters):
        image = imageio.v2.imread(fr'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\images\img_{iter}.png')
        frames.append(image)

    imageio.mimsave(fr'C:\Users\jthan\OneDrive\Desktop\2023\PP\RoadTripOptimizer\images\test.gif',
                    frames,          
                    fps = 5)


def main():
    Birmingham = (33.5186, -86.8104)
    Tucson = (32.1545, -110.8782)
    Little_Rock = (34.7465, -92.2896)
    San_Francisco = (37.7749, -122.4194)
    Boulder = (40.01499, -105.27055)
    hartford = (41.7658, -72.6734)
    Wilmington = (39.7459, -75.5466)
    Miami = (25.7617, -80.1918)
    Athens = (33.9519, -83.3576)
    Boise = (43.6150, -116.2023)
    Chicago = (41.8781, -87.6298)
    Lafayette = (30.2241, -92.0198)
    Iowa_City = (41.6611, -91.5302)
    Lawrence = (38.9717, -95.2353)
    Louisville = (38.2527, -85.7585)
    New_Orleans = (29.9511, -90.0715)
    Portland = (45.5051, -122.6750)
    Annapolis = (38.9784, -76.4922)
    boston = (42.3601, -71.0589)
    Ann_Arbor = (42.2808, -83.7430)
    Duluth = (46.7867, -92.1005)
    Gulfport = (30.3674, -89.0928)
    Kansas_City = (39.0997, -94.5786)
    Livingston = (45.6629, -110.5600)
    Omaha = (41.2565, -95.9345)
    Reno = (39.5296, -119.8138)
    Portsmouth = (43.0718, -70.7626)
    Ocean_City = (38.3365, -75.0849)
    Santa_Fe = (35.6870, -105.9378)
    New_York_City = (40.7128, -74.0060)
    Asheville = (35.5951, -82.5515)
    Fargo = (46.8772, -96.7898)
    Cincinnati = (39.1031, -84.5120)
    Tulsa = (36.1540, -95.9928)
    Portland = (43.6615, -70.2553)
    Philadelphia = (39.9526, -75.1652)
    Newport = (41.4901, -71.3128)
    Folly_Beach = (32.6552, -79.9404)
    Sioux_Falls = (43.5446, -96.7311)
    Nashville = (36.1627, -86.7816)
    Austin = (30.2672, -97.7431)
    Park_City = (40.6461, -111.4979)
    Burlington = (44.4759, -73.2121)
    Arlington = (38.8799697, -77.1067698)
    Seattle = (47.6062, -122.3321)
    Morgantown = (39.6295, -79.9559)
    Madison = (43.0731, -89.4012)
    Jackson_Hole = (43.4799, -110.7624)

    Cities = [Birmingham, Tucson, Little_Rock, San_Francisco, Boulder, hartford, Wilmington, Miami, Athens, Boise, Chicago, Lafayette, Iowa_City, Lawrence, Louisville, New_Orleans, Portland, Annapolis, boston, Ann_Arbor, Duluth, Gulfport, Kansas_City, Livingston, Omaha, Reno, Portsmouth, Ocean_City, Santa_Fe, New_York_City, Asheville, Fargo, Cincinnati, Tulsa, Portland, Philadelphia, Newport, Folly_Beach, Sioux_Falls, Nashville, Austin, Park_City, Burlington, Arlington, Seattle, Morgantown, Madison, Jackson_Hole]

    distances = calc_distance(Cities)

    plot_cities(Cities, True)

    
if __name__ == "__main__":
    main()

