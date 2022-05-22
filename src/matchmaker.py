import operator
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import timedelta, date, datetime
from dataclasses import dataclass
from typing import List, Dict, Callable, Union, Any
from src.data_reader import RawData, ProductType, Distance
from definitions import config_path, data_path, log_path, out_path


logging.basicConfig(filename=log_path / "matchmaker.log", level=logging.DEBUG, filemode='w',
                    format='%(asctime)s: %(message)s')


@dataclass
class Weights:
    views: int
    messages: int
    contacts: int


class Config:
    """Loads configuration from yaml.
    Does some initial tasks, like filtering customers based on the config, and setting weights.
    Provides interface to access some values more easily.
    """

    def __init__(self, config_path_: Path = config_path):
        try:
            with open(config_path_, "r") as cp:
                self.config_data = yaml.safe_load(cp)
                logging.info(f"Loaded configuration from {config_path_}.")
        except:
            logging.error(f"Reading of {config_path_} failed. Falling back on defaults.")
            self.config_data: Dict[str, Union[str, int, Dict[str, Any]]] = {
                "source": "source.xlsx",
                "suggestions": 5,
                "activity_weight": {
                    "views": 1,
                    "messages": 1,
                    "contacts": 1},
                "penalty": {
                    "quantity": 0,
                    "shipping": 0,
                    "distance": 0,
                    "pricing": 0},
                "customer_filter": {
                    "blocked": False,
                    "deleted": False,
                    "selling": False},
                "time_window": {
                    "before": '2021-11-29',
                    "days": 7}}

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

    def get_until_date(self) -> date:
        return datetime.fromisoformat(self.config_data['time_window']['before'])

    def get_penalties(self) -> Dict[str, float]:
        return self.config_data['penalty']

    def get_suggestion_count(self) -> int:
        return self.config_data['suggestions']


class MatchMaker:
    """Contains the processing and logic of the match selection."""

    def __init__(self, config_path_: Path = config_path, write: bool = True):
        """
        Create MatchMaker object with the particular provided configuration.

        The object creation involves the full process of data reading, processing, and
        suggestion computation and so it may take several seconds.
        :param config_path_: path to program configuration yaml file
        :param write: whether to write output or not
        """
        self.config: Config = Config(config_path_)
        xlspath: Path = data_path / self.config.config_data['source']
        _weights: Weights = self.config.get_weights()
        self.raw: RawData = RawData(xlspath)
        logging.info(f"Data from file {xlspath} read and pre-processed.")
        self.customers: pd.DataFrame = self.config.filter_customers(self.raw.proc_customers)
        self.products: pd.DataFrame = self.raw.proc_products
        self.distance: Distance = self.raw.distance
        self.customer_preferences: Dict[int, pd.DataFrame] = self.get_preferences()
        self.triplet_preferences: Dict[int, pd.DataFrame] = self.aggregate_preferences()
        logging.info("Customer preferences processing finished.")
        self.last_products: pd.DataFrame = self.get_last_products()
        self.suggestions: Dict[int, List[int]] = self.match_last_products()
        logging.info("Generation of matches for relevant customers finished.")
        if write:
            self.store_results()

    def store_results(self, tp: int = 3) -> None:
        """For each customer, write the top preferences and the top recommendations.

        Note that the recommendations list one item per each category-sub_category-product_type
        combination. Therefore, while 5 products are recommended, there can be more than 5 items."""
        tops: Dict[int, pd.DataFrame] = {}
        for cid, cpref in self.triplet_preferences.items():
            top_pref: pd.Series = cpref.nlargest(tp, 'weight')[['category', 'sub_category', 'product_type']]\
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
        logging.info(f"The results were written to {out_path / 'results.xlsx'}.")

    def match_last_products(self) -> Dict[int, List[int]]:
        """Matche products added in the desired time window with individual customer preferences (triplet preference).

        If the triplet (category-sub_category-product_type) doesn't yield the required count of matches, repeats the
        process, ignoring the sub_category."""
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
        logging.info(f"Recommendations for {len(matches)} customers were generated.")
        return matches

    def _score_product_affordability(self, cid: int, cpref: pd.DataFrame, cid_ps: pd.Series,
                                     merge_on: List[str]) -> pd.DataFrame:
        """For products selected based on customer activity-based preferences, score them based on the general
        affordability, giving penalties for low amount, absence of shipping, large distance, and potentially too
        elevated price."""
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
        """Extract products added in the desired time window, which were published and weren't yet sold."""
        since_time: date = self.config.get_since_date()
        last_products: pd.DataFrame = self.products[(self.products['created'] >= since_time)
                                                    & (self.config.get_until_date() >= self.products['created'])
                                                    & (self.products['sold'] == False)
                                                    & (self.products['published'] == True)]
        last_products['product_type'] = last_products['product_type'].apply(ProductType.to_list)
        last_products = last_products.explode('product_type')
        last_products = last_products.merge(self.raw.mean_prices, how='left',
                                            on=['category', 'sub_category', 'product_type'])
        logging.info(f"List of products added between {self.config.get_since_date()} and {self.config.get_until_date()}"
                     f" was extracted, counting {len(last_products.index)}.")
        return last_products

    def aggregate_preferences(self) -> Dict[int, pd.DataFrame]:
        """Transform preferences to account for individual product type (which are mentions only as sum).
        This creates another column, where product_type is unpacked, and everything else copied across the new lines.
        Finally all possible triples combination (category-sub_category-product_type) are grouped by, to obtain
        user detailed interest."""
        triplet_preferences: Dict[int, pd.DataFrame] = {}
        for cid, pref_df in self.customer_preferences.items():
            pref_df['product_type'] = pref_df['product_type'].apply(ProductType.to_list)
            exp_pref = pref_df.explode('product_type').drop(columns=['product_id'], inplace=False)
            agg_pref: pd.DataFrame = exp_pref.groupby(['category', 'sub_category', 'product_type']).sum().reset_index()
            triplet_preferences[cid] = agg_pref.sort_values(by=['weight'], ascending=False)
        logging.info(f"Preferences with weights for activity based category-sub_category-product_type triplets "
                     f"were generated for each customer, where possible.")
        return triplet_preferences

    def get_preferences(self) -> Dict[int, pd.DataFrame]:
        """Get preferences for each customer based on their activities and activity score."""
        customer_preferences: Dict[int, pd.DataFrame] = {}
        self.customers['aggregate_activ'] = self.customers \
            .apply(lambda row: self._gather_activities(row.message, row.contact, row.pview, self.config.get_weights()),
                   axis=1)
        for row in self.customers[['customer_id', 'aggregate_activ']].itertuples(index=False):
            customer_preferences[row.customer_id] = self._prod_id_to_type(row.aggregate_activ)
        logging.info(f"Customer preference data created.")
        return customer_preferences

    def _prod_id_to_type(self, pids_w_weights: Dict[int, float]) -> pd.DataFrame:
        """Assign product given category combination a precalculated weight value, and return
        a DataFrame with relevant scored product categorization."""
        category: pd.DataFrame = self.products.loc[self.products['product_id'].isin(pids_w_weights.keys()),
                                                   ['product_id', 'category', 'sub_category', 'product_type']]
        category['weight'] = category['product_id'].apply(pids_w_weights.__getitem__)
        return category

    @staticmethod
    def _gather_activities(message: List[int], contact: List[int], view: List[int],
                           _weights: Weights, combinator: Callable[[float, float], float] = operator.add) \
            -> Dict[int, float]:
        """Combine activities upon a given product into an aggregate activity score (weights), over PageViews, Message,
        and Contact view. The combinator defines how to choose score, if several activities were performed on a single
        product id (summing is default)."""
        umsg: Dict[int, int] = {m: _weights.messages for m in message}
        ucnt: Dict[int, int] = {c: _weights.contacts for c in contact}
        uview: Dict[int, int] = {v: _weights.views for v in view}

        acts: Dict[int, float] = {}
        for sets in [umsg, ucnt, uview]:
            acts = {pid: combinator(acts.get(pid, 0), sets.get(pid, 0)) for pid in set(acts) | set(sets)}
        return acts


if __name__ == "__main__":
    mm: MatchMaker = MatchMaker()
