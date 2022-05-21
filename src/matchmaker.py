import operator
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple
from src.data_reader import RawData, Trade, ProductType
import logging
from definitions import datafile_path
from pathlib import Path
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import timedelta, date, datetime


loc = Nominatim(user_agent="GetLoc")


@dataclass
class Weights:
    views: int
    messages: int
    contacts: int


class MatchMaker:

    def __init__(self, xlspath: Path, _weights: Weights):
        self.raw: RawData = RawData(xlspath)
        self.customers: pd.DataFrame = self.raw.proc_customers
        self.products: pd.DataFrame = self.raw.proc_products
        logging.info(f"Data from file {xlspath} read")
        self.customer_preferences: Dict[int, pd.DataFrame] = self.get_preferences(_weights)
        self.triplet_preferences: Dict[int, pd.DataFrame] = self.aggregate_preferences()
        logging.info("Extracted customer preferences")
        self.last_products: pd.DataFrame = self.get_last_products()
        self.match_last_products()

    def match_last_products(self, count: int = 10, noshipping_penalty: int = 0.5, distance_penalty: int = 0.1) \
            -> Dict[int, List[int]]:
        matches: Dict[int, List[int]] = {}
        for cid, cpref in self.triplet_preferences.items():
            cid_ps: pd.Series = self.customers.loc[self.customers['customer_id'] == cid,
                                                   ['country', 'message', 'amount']].iloc[0]
            match: pd.DataFrame = self.last_products.merge(cpref, how='inner',
                                                           on=['category', 'sub_category', 'product_type'])
            match = match[~match['product_id'].isin(cid_ps['message'])]
            if cid_ps['amount'] is not None:
                low: float = cid_ps['amount'][0]
                match['weight'] = match[['quantity', 'weight', 'unit']]\
                    .apply(lambda row: row.weight + (0.5 if row.quantity >= low and row.unit == 't' else 0), axis=1)
            if noshipping_penalty > 0:
                match['weight'] = match[['weight', 'shipping_available']]\
                    .apply(lambda row: row.weight - (0 if row.shipping_available else noshipping_penalty), axis=1)
            if distance_penalty > 0:
                customer_gps: Tuple[float, float] = self._get_lat_lon(cid_ps['country'])
                match['distance'] = match['country']\
                    .apply(lambda country: geodesic(self._get_lat_lon(country), customer_gps).kilometers)
                match['weight'] = match[['weight', 'distance']]\
                    .apply(lambda row: row.weight - row.distance/1000 * distance_penalty, axis=1)
            matches[cid] = match['product_id'].to_list()
        return matches

    @staticmethod
    def _get_lat_lon(country: str) -> Tuple[float, float]:
        gps = loc.geocode(country)
        return gps.latitude, gps.longitude

    def get_last_products(self, period: timedelta = timedelta(days=7),
                          last_date: datetime = datetime.fromisoformat('2021-11-29')) -> pd.DataFrame:
        since_time: date = last_date - period
        last_products: pd.DataFrame =  self.products[  (self.products['created'] >= since_time)
                                                     & (self.products['sold'] == False)
                                                     & (self.products['published'] == True)]
        last_products['product_type'] = last_products['product_type'].apply(ProductType.to_list)
        return last_products.explode('product_type')

    def aggregate_preferences(self) -> Dict[int, pd.DataFrame]:
        triplet_preferences: Dict[int, pd.DataFrame] = {}
        for cid, pref_df in self.customer_preferences.items():
            pref_df['product_type'] = pref_df['product_type'].apply(ProductType.to_list)
            exp_pref = pref_df.explode('product_type').drop(columns=['product_id'], inplace=False)
            agg_pref: pd.DataFrame = exp_pref.groupby(['category', 'sub_category', 'product_type']).sum().reset_index()
            triplet_preferences[cid] = agg_pref.sort_values(by=['weight'], ascending=False)
        return triplet_preferences

    def get_preferences(self, _weights: Weights) -> Dict[int, pd.DataFrame]:
        customer_preferences: Dict[int, pd.DataFrame] = {}
        self.customers['aggregate_activ'] = self.customers \
            .apply(lambda row: self._gather_activities(row.message, row.contact, row.pview, _weights), axis=1)
        for row in self.customers[['customer_id', 'aggregate_activ']].itertuples(index=False):
            customer_preferences[row.customer_id] = self._prod_id_to_type(row.aggregate_activ)
        logging.info(f"Customer preference data created.")
        return customer_preferences

    def _prod_id_to_type(self, pids_w_weights: Dict[int, float]) -> pd.DataFrame:
        logging.info(f"Converting product ids to product categories.")
        category: pd.DataFrame = self.products.loc[self.products['product_id']
                                                       .isin(pids_w_weights.keys()), ['product_id', 'category',
                                                                                      'sub_category', 'product_type']]
        category['weight'] = category['product_id'].apply(pids_w_weights.__getitem__)
        return category

    @staticmethod
    def _gather_activities(message: List[int], contact: List[int], view: List[int],
                           _weights: Weights, combinator: Callable[[float, float], float] = operator.add) \
            -> Dict[int, float]:
        logging.info(f"Gathering weighted activities.")
        umsg: Dict[int, int] = {m: _weights.messages for m in message}
        ucnt: Dict[int, int] = {c: _weights.contacts for c in contact}
        uview: Dict[int, int] = {v: _weights.views for v in view}

        acts: Dict[int, float] = {}
        for sets in [umsg, ucnt, uview]:
            acts = {pid: combinator(acts.get(pid, 0), sets.get(pid, 0)) for pid in set(acts) | set(sets)}
        return acts


if __name__ == "__main__":
    weights: Weights = Weights(1, 5, 5)

    mm: MatchMaker = MatchMaker(datafile_path, weights)
    pass
