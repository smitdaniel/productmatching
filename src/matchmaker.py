import operator
from dataclasses import dataclass
from typing import List, Dict, Callable, Union, Any
from src.data_reader import RawData, ProductType, Distance
import logging
import yaml
from definitions import config_path, data_path, log_path, out_path
from pathlib import Path
import pandas as pd
from datetime import timedelta, date, datetime


logging.basicConfig(filename=log_path / "reader.log", level=logging.DEBUG, filemode='w',
                    format='%(asctime)s: %(message)s')


@dataclass
class Weights:
    views: int
    messages: int
    contacts: int


class Config:
    """Loads configuration from yaml.
    Does some initial tasks, like filtering customers based on the config, and setting weights.
    """

    def __init__(self, config_path_: Path = config_path):
        with open(config_path_, "r") as cp:
            try:
                self.config_data = yaml.safe_load(cp)
                logging.info(f"Loaded configuration from {cp}.")
            except yaml.YAMLError as yr:
                logging.error(yr)
                # TODO: add default configuration

    def filter_customers(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        if not any(self.config_data['customer_filter'].values()):
            return customer_df
        else:
            pick: Callable[[str, pd.DataFrame], Union[pd.DataFrame, bool]] = \
                lambda filter_type, filter_df: filter_df if self.config_data['customer_filter'][filter_type] else True
            not_blocked = pick('blocked', customer_df['blocked'] == False)
            not_deleted = pick('deleted', customer_df['deleted'] == False)
            not_seller = pick('selling', customer_df['interest'].apply(lambda trade: trade.buy | trade.other))

            return customer_df[not_deleted & not_blocked & not_seller]

    def get_weights(self) -> Weights:
        w: Dict[str, int] = self.config_data['activity_weight']
        return Weights(w['views'], w['messages'], w['contacts'])

    def get_since_date(self) -> date:
        tw: Dict[str, Any] = self.config_data['time_window']
        period: timedelta = timedelta(days=tw['days'])
        last_date: datetime = datetime.fromisoformat(tw['before'])
        return last_date - period

    def get_penalties(self) -> Dict[str, float]:
        return self.config_data['penalty']

    def get_suggestion_count(self) -> int:
        return self.config_data['suggestions']


class MatchMaker:

    def __init__(self, config_path_: Path = config_path):
        self.config: Config = Config(config_path_)
        xlspath: Path = data_path / self.config.config_data['source']
        _weights: Weights = self.config.get_weights()
        self.raw: RawData = RawData(xlspath)
        logging.info(f"Data from file {xlspath} read")
        self.customers: pd.DataFrame = self.config.filter_customers(self.raw.proc_customers)
        self.products: pd.DataFrame = self.raw.proc_products
        self.distance: Distance = self.raw.distance
        self.customer_preferences: Dict[int, pd.DataFrame] = self.get_preferences()
        self.triplet_preferences: Dict[int, pd.DataFrame] = self.aggregate_preferences()
        logging.info("Extracted customer preferences")
        self.last_products: pd.DataFrame = self.get_last_products()
        self.suggestions: Dict[int, List[int]] = self.match_last_products()
        logging.info("Generated matches for relevant customers.")
        self.store_results()

    def store_results(self) -> None:
        tops: Dict[int, pd.DataFrame] = {}
        for cid, cpref in self.triplet_preferences.items():
            top_pref: pd.Series = cpref.nlargest(3, 'weight')[['category', 'sub_category', 'product_type']]\
                .apply(lambda row: "-".join([row.category, row.sub_category, row.product_type]), axis=1)
            top_products: pd.Series = self.last_products\
                .loc[self.last_products['product_id'].isin(self.suggestions[cid]), ['product_id', 'category',
                                                                                    'sub_category', 'product_type']]\
                .apply(lambda row: str(row.product_id) + ": " +
                                   '-'.join([row.category, row.sub_category, row.product_type]), axis=1)
            if len(top_products.index) == 0:
                top_products = pd.Series([f"None in searched {self.config.config_data['time_window']['days']} days."])
            tops[cid]: pd.DataFrame = pd.DataFrame({'preferences': top_pref.reset_index(drop=True),
                                                    'suggestions': top_products.reset_index(drop=True)})
        tops_df: pd.DataFrame = pd.concat(tops)
        tops_df.index.names = ["customer id", None]
        tops_df.to_excel(out_path / "results.xlsx")

    def match_last_products(self) -> Dict[int, List[int]]:
        matches: Dict[int, List[int]] = {}
        for cid, cpref in self.triplet_preferences.items():
            wider_match_list: List[int] = list()
            cid_ps: pd.Series = self.customers.loc[self.customers['customer_id'] == cid,
                                                   ['country', 'message', 'amount']].iloc[0]
            match: pd.DataFrame = self._score_product_affordability(cid, cpref, cid_ps,
                                                                    merge_on=['category', 'sub_category',
                                                                              'product_type'])
            # check if enough specific results were produced
            short_hits: int = self.config.get_suggestion_count() - len(match.index)
            if short_hits > 0:  # add more generic suggestions
                wider_match: pd.DataFrame = self._score_product_affordability(cid, cpref, cid_ps,
                                                                              merge_on=['category', 'product_type'])
                wider_match = wider_match[~wider_match['product_id'].isin(match['product_id'])]
                wider_match_list = wider_match.nlargest(short_hits, 'weight')['product_id'].to_list()
            matches[cid] = match['product_id'].to_list() + wider_match_list
        return matches

    def _score_product_affordability(self, cid: int, cpref: pd.DataFrame, cid_ps: pd.Series,
                                     merge_on: List[str]) -> pd.DataFrame:
        penalties: Dict[str, float] = self.config.get_penalties()
        match: pd.DataFrame = self.last_products.merge(cpref, how='inner', on=merge_on)
        match = match[(~match['product_id'].isin(cid_ps['message'])) & (match['customer_id'] != cid)]
        if len(match.index) == 0:
            return match
        if cid_ps['amount'] is not None and penalties['quantity'] > 0:
            low: float = cid_ps['amount'][0]
            match['weight'] = match[['quantity', 'weight', 'unit']] \
                .apply(lambda row: row.weight - (penalties['quantity'] if row.quantity < low and row.unit == 't'
                                                 else 0), axis=1)
        if penalties['shipping'] > 0:
            match['weight'] = match[['weight', 'shipping_available']] \
                .apply(lambda row: row.weight - (0 if row.shipping_available else penalties['shipping']), axis=1)
        if penalties['distance'] > 0:
            match['distance'] = match['country'] \
                .apply(lambda country: self.distance.get(country, cid_ps['country']))
            match['weight'] = match[['weight', 'distance']] \
                .apply(lambda row: row.weight - row.distance * penalties['distance'], axis=1)
        if penalties['pricing'] > 0:
            match['weight'] = match[['weight', 'eur_price', 'mean_price']] \
                .apply(lambda row: row.weight - (row.eur_price - row.mean_price) * penalties['pricing'], axis=1)
        return match.nlargest(self.config.get_suggestion_count(), 'weight')

    def get_last_products(self) -> pd.DataFrame:
        since_time: date = self.config.get_since_date()
        last_products: pd.DataFrame = self.products[(self.products['created'] >= since_time)
                                                    & (self.config.config_data['time_window']['before'] >=
                                                       self.products['created'])
                                                    & (self.products['sold'] == False)
                                                    & (self.products['published'] == True)]
        last_products['product_type'] = last_products['product_type'].apply(ProductType.to_list)
        last_products = last_products.explode('product_type')
        last_products = last_products.merge(self.raw.mean_prices, how='left',
                                            on=['category', 'sub_category', 'product_type'])
        return last_products

    def aggregate_preferences(self) -> Dict[int, pd.DataFrame]:
        triplet_preferences: Dict[int, pd.DataFrame] = {}
        for cid, pref_df in self.customer_preferences.items():
            pref_df['product_type'] = pref_df['product_type'].apply(ProductType.to_list)
            exp_pref = pref_df.explode('product_type').drop(columns=['product_id'], inplace=False)
            agg_pref: pd.DataFrame = exp_pref.groupby(['category', 'sub_category', 'product_type']).sum().reset_index()
            triplet_preferences[cid] = agg_pref.sort_values(by=['weight'], ascending=False)
        return triplet_preferences

    def get_preferences(self) -> Dict[int, pd.DataFrame]:
        customer_preferences: Dict[int, pd.DataFrame] = {}
        self.customers['aggregate_activ'] = self.customers \
            .apply(lambda row: self._gather_activities(row.message, row.contact, row.pview, self.config.get_weights()),
                   axis=1)
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
    mm: MatchMaker = MatchMaker(config_path)
    pass
