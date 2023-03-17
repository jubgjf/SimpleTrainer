import argparse
from abc import abstractmethod
from typing import Any


class Config:
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def _get_var_name(var: Any) -> str:
        return [k for k, v in locals().items() if v == var][0]

    def argparse(self):
        parser = argparse.ArgumentParser()
        for k, v in self.__dict__.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        args = parser.parse_args()

        args = vars(args)
        for k in self.__dict__.keys():
            setattr(self, k, args[k])

    def dump_config(self):
        max_key_str_len = max([len(str(l)) for l in self.__dict__.keys()])
        max_value_str_len = max([len(str(l)) for l in self.__dict__.values()])

        print("╔" + "═" * (max_key_str_len + 6 + max_value_str_len) + "╗")
        for k, v in self.__dict__.items():
            padding_left_ws = " " * (max_key_str_len - len(str(k)))
            padding_right_ws = " " * (max_value_str_len - len(str(v)))
            print(f"║ {k}{padding_left_ws} => {v}{padding_right_ws} ║")
        print("╚" + "═" * (max_key_str_len + 6 + max_value_str_len) + "╝")
