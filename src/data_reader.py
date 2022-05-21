from dataclasses import dataclass
from functools import partial
import pandas as pd
import logging
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from definitions import log_path, datafile_path

logging.basicConfig(filename=log_path / "reader.log", level=logging.DEBUG)


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
        self.proc_products = self._process_products()
        self.proc_customers = self._process_customers()

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
        customer['s-b-o'] = customer['interest'].apply(lambda i: Trade('sell' in i, 'buy' in i, 'other' in i))
        customer['amount'] = customer['waste_amount'].apply(self._to_range)
        customer.drop(columns=['interest', 'waste_amount', 'created'], inplace=True)
        customer[['message', 'contact', 'pview']] = customer['customer_id'].apply(self._get_activity)
        return customer

    def _get_activity(self, cid: int) -> pd.Series:
        messages: List[int] = self._interaction_to_list(self.connect, cid, 'message_sent')
        contacts: List[int] = self._interaction_to_list(self.connect, cid, 'contact_show')
        views: List[int] = self._interaction_to_list(self.activity, cid, 'event_name')
        return pd.Series([messages, contacts, views])

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
            return pd.Series([in_price/1000, in_qty/1000, 't'])


if __name__ == '__main__':

    rd = RawData(datafile_path)
    pass
