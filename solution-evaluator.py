import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAVEL_TIME_TOLERANCE_SECS = 3 #feasibility tolerance
SOLU_FILE = 'output.csv'
INPUT_FILE = 'Data/optimization_take_home.csv'

solu = pd.read_csv(SOLU_FILE)
deliveries = pd.read_csv(INPUT_FILE, parse_dates=['created_at', 'food_ready_time'])
merged = pd.merge(left=solu, right=deliveries, left_on='Delivery ID', right_on='delivery_id')
merged['Route Point Type'] = merged['Route Point Type'].apply(lambda x: x.strip())

merged['rp_lat'] = np.where(merged['Route Point Type']=='Pickup', merged['pickup_lat'], merged['dropoff_lat'])
merged['rp_long'] = np.where(merged['Route Point Type']=='Pickup', merged['pickup_long'], merged['dropoff_long'])
merged['rp_time'] = pd.to_datetime(merged['Route Point Time'], unit='s')
merged.sort_values(['Route ID', 'Route Point Index'], inplace=True)

from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def get_travel_seconds(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2)*1000/4.5


def validate_travel_time_and_sequence(merged):
    for _, subdf in merged.groupby('Route ID'):
        subdf = subdf.sort_values('Route Point Index')
        pre_lat, pre_lon, pre_time= None, None, None
        added_delivery = set()
        for _, row in subdf.iterrows():

            if pre_lat is not None:
                travel_seconds = get_travel_seconds(pre_lat, pre_lon, row['rp_lat'], row['rp_long'])
                #verify travel time
                assert (row['rp_time'] - pre_time).total_seconds() \
                            >= np.floor(travel_seconds) - TRAVEL_TIME_TOLERANCE_SECS, \
                        """not enough travel time to {}
                            expected travel seconds {}, got {} in solution
                        """.format(row, travel_seconds, (row['rp_time'] - pre_time).total_seconds())


            if row['Route Point Type'] == 'Pickup':
                added_delivery.add(row['delivery_id'])
            elif row['Route Point Type'] == 'DropOff':
                added_delivery.remove(row['delivery_id'])
            else:
                print("error route type '{}' not recognized".format(row['Route Point Type']))
            pre_lat, pre_lon, pre_time = row['rp_lat'], row['rp_long'], row['rp_time']

        assert len(added_delivery)==0, "not all deliveries pickuped up are dropped off"

def validate_delivery_time(merged):
    def get_asap(delivery_df):
        return (delivery_df.rp_time.max() - delivery_df.created_at.min()).total_seconds()
    asap = merged.groupby('delivery_id').apply(get_asap).mean()/60.0
    assert asap <= 45, "average delivery time {} > 45 min".format(asap)


assert len(deliveries)*2 == len(solu), \
    """there are {} deliveries, expecting {} route pints
        but got {} route points in solu""".format(len(deliveries), len(deliveries)*2, len(solu))

assert len(deliveries)*2 == len(solu[['Delivery ID', 'Route Point Type']].drop_duplicates()), \
    """there are {} deliveries, expecting {} route pints
        but got {} route points in solu""".format(len(deliveries), len(deliveries)*2,
                                                  len(solu[['Delivery ID', 'Route Point Type']].drop_duplicates()))

assert sorted(deliveries.delivery_id.unique()) == sorted(solu['Delivery ID'].unique()), \
    """deliveries missing: """.format(np.setdiff1d(deliveries.delivery_id.unique(), solu['Delivery ID'].unique()))

assert (merged['rp_time'] >= merged['food_ready_time']).all(), \
    "not all pickup time are after the food ready time"
validate_travel_time_and_sequence(merged)
validate_delivery_time(merged)


def get_dat(route_df):
    return (route_df.rp_time.max() - pd.to_datetime('2002-03-15 02:00:00')).total_seconds()
efficiency = len(deliveries)/(merged.groupby('Route ID').apply(get_dat).sum()/3600.0)
print(u"the efficiency is {}".format(efficiency))

