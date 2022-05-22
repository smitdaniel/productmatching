from dataclasses import dataclass
from functools import partial
import pandas as pd
import logging
import json
from typing import List, Dict, Optional, Union, Tuple, Set
from pathlib import Path
from definitions import distance_cache_path
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

loc = Nominatim(user_agent="MatchMaker")


@dataclass
class Trade:
    sell: bool
    buy: bool
    other: bool


@dataclass
class ProductType:
    waste: bool
    byproduct: bool
    recycled: bool

    def to_list(self) -> List[str]:
        pt_list: List[str] = []
        for pt in ['waste', 'byproduct', 'recycled']:
            if self.__getattribute__(pt):
                pt_list.append(pt)
        return pt_list


class Distance:

    def __init__(self, products_: pd.Series, customers_: pd.Series):
        self.all_countries: Set[str] = set(pd.concat([products_, customers_], ignore_index=True))
        self.matrix: Dict[str, Dict[str, float]] = {}
        if distance_cache_path.is_file():
            with open(distance_cache_path, 'r') as matrix_file:
                try:
                    self.matrix = json.load(matrix_file)
                except json.JSONDecodeError as je:
                    logging.error(je)
        if not self.all_countries.issubset(self.matrix.keys()):
            logging.info(f"Distance matrix not found or incomplete, updating.")
            self._update_matrix()

    def _update_matrix(self) -> None:
        countries_w_gps: Dict[str, Tuple[float, float]] = {c: self._get_lat_lon(c) for c in self.all_countries}
        self.matrix = {cl: {cr: 0 for cr in countries_w_gps} for cl in countries_w_gps}
        for country_l, gps_l in countries_w_gps.items():
            for country_r, gps_r in countries_w_gps.items():
                if country_r == country_l:
                    continue
                elif self.matrix[country_r][country_l] != 0:
                    self.matrix[country_l][country_r] = self.matrix[country_r][country_l]
                else:
                    self.matrix[country_l][country_r] = geodesic(gps_l, gps_r).kilometers / 1000
        with open(distance_cache_path, 'w') as matrix_file:
            json.dump(self.matrix, matrix_file)

    def get(self, country_l: str, country_r: str) -> float:
        return self.matrix[country_l][country_r]

    @staticmethod
    def _get_lat_lon(country: str) -> Tuple[float, float]:
        gps = loc.geocode(country)
        return gps.latitude, gps.longitude


class RawData:

    exp_sheets: Dict[str, Union[List[str], bool]] = {
        'data_dictionary': False,
        'products': ['created_datetime'],
        'customers': ['created'],
        'connect': ['created'],
        'activity': ['created']}
    curr_to_eur: Dict[str, float] = {"PLN": 0.25, "CZK": 0.04, "EUR": 1, "USD": 0.95}
    exp_currencies: List[str] = curr_to_eur.keys()

    def __init__(self, xlspath: Path):
        if not xlspath.is_file():
            raise ValueError(f"The requested file {xlspath} does not exist!")
        else:
            for sheet, datetime in self.exp_sheets.items():
                setattr(self, sheet, pd.read_excel(xlspath, sheet_name=sheet, parse_dates=datetime))
                logging.info(f"Reading sheet {sheet}")
        self.proc_products: pd.DataFrame = self._process_products()
        self.proc_customers: pd.DataFrame = self._process_customers()
        self.distance: Distance = Distance(self.proc_products['country'], self.proc_customers['country'])
        self.mean_prices: pd.DataFrame = self._get_mean_price()

    def _process_products(self) -> pd.DataFrame:
        product: pd.DataFrame = self.products.copy()
        product.rename(columns={'cyrkl_product_id': 'product_id', 'cyrkl_contact_id': 'customer_id',
                                'created_datetime': 'created'}, inplace=True)
        product['price'] = product[['price']].apply(partial(pd.to_numeric, errors='coerce'))
        product['waste_code'] = product[['waste_code']].astype('Int64')
        product['eur_price'] = product.loc[:, ['price', 'currency']]\
            .apply(lambda row: self._price_conversion(row.price, row.currency), axis=1)
        product[['eur_price', 'quantity', 'unit']] = product.loc[:, ['eur_price', 'quantity', 'unit']]\
            .apply(lambda row: self._kg_to_ton(row.eur_price, row.quantity, row.unit), axis=1)
        product['product_type'] = product['product_type']\
            .apply(lambda lbl: ProductType('waste' in lbl, 'by-product' in lbl, 'recycled' in lbl))
        product.drop(columns=['price', 'currency', 'name', 'description'], inplace=True)
        return product

    def _process_customers(self) -> pd.DataFrame:
        customer: pd.DataFrame = self.customers.copy()
        customer['interest'] = customer['interest'].apply(lambda i: Trade('sell' in i, 'buy' in i, 'other' in i))
        customer['amount'] = customer['waste_amount'].apply(self._to_range)
        customer.drop(columns=['waste_amount', 'created'], inplace=True)
        customer[['message', 'contact', 'pview']] = customer['customer_id'].apply(self._get_activity)
        return customer

    def _get_activity(self, cid: int) -> pd.Series:
        messages: List[int] = self._interaction_to_list(self.connect, cid, 'message_sent')
        contacts: List[int] = self._interaction_to_list(self.connect, cid, 'contact_show')
        views: List[int] = self._interaction_to_list(self.activity, cid, 'event_name')
        return pd.Series([messages, contacts, views])

    def _get_mean_price(self) -> pd.DataFrame:
        priced_products: pd.DataFrame = self.proc_products[['category', 'sub_category', 'product_type', 'eur_price',
                                                  'pricing_type', 'unit']].copy()
        priced_products = priced_products[(priced_products['pricing_type'] == 'per_unit')
                                          & (priced_products['unit'] == 't')]
        priced_products.drop(columns=['pricing_type', 'unit'], inplace=True)
        priced_products['product_type'] = priced_products['product_type'].apply(ProductType.to_list)
        priced_products = priced_products.explode('product_type')
        priced_products = priced_products.groupby(by=['category', 'sub_category', 'product_type']).mean().reset_index()
        priced_products.rename(columns={'eur_price': 'mean_price'}, inplace=True)
        return priced_products

    @staticmethod
    def _to_range(rng_lbl: Optional[str]) -> Optional[Tuple[float, float]]:
        if pd.isna(rng_lbl):
            return None
        elif rng_lbl == 'under_100_t':
            return 0, 100
        elif rng_lbl == 'between_100_and_400_t':
            return 100, 400
        elif rng_lbl == 'over_400_t':
            return 400, float('inf')
        else:
            raise ValueError(f"Unexpected waste amount indication of {rng_lbl}.")

    @staticmethod
    def _interaction_to_list(parse_df: pd.DataFrame, cid: int, activity: str) -> List[int]:
        all_activity: Dict[str, str] = {
            'message_sent': 'Yes',
            'contact_show': 'Yes',
            'event_name': 'PageView'}
        val = all_activity[activity]
        if activity not in all_activity.keys():
            raise ValueError(f"Argument activity must be one of {all_activity}, You passed {activity}.")
        return parse_df.loc[(parse_df['customer_id'] == cid) & (parse_df[activity] == val), 'product_id'].to_list()

    @classmethod
    def _price_conversion(cls, value: float, in_currency: str, out_currency: str = 'EUR') -> float:
        if in_currency not in cls.exp_currencies or out_currency not in cls.exp_currencies:
            raise ValueError(f"Allowed currencies are {cls.exp_currencies}, "
                             f"You passed {in_currency} and {out_currency}.")
        val_in_eur: float = value * cls.curr_to_eur[in_currency]
        val_in_out: float = val_in_eur / cls.curr_to_eur[out_currency]

        return val_in_out

    @classmethod
    def _kg_to_ton(cls, in_price: float, in_qty: float, in_unit: str) -> pd.Series:
        if in_unit != 'kg':
            return pd.Series([in_price, in_qty, in_unit])
        else:
            return pd.Series([in_price * 1000, in_qty / 1000, 't'])
