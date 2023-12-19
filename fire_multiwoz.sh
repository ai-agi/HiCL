#!/bin/bash
python main_multiwoz.py --target_domain="Book_Hotel"
python main_multiwoz.py --target_domain="Book_Hotel" --run_mode="test"
python main_multiwoz.py --target_domain="Book_Restaurant"
python main_multiwoz.py --target_domain="Book_Restaurant" --run_mode="test"
python main_multiwoz.py --target_domain="Book_Train"
python main_multiwoz.py --target_domain="Book_Train" --run_mode="test"
python main_multiwoz.py --target_domain="Find_Attraction"
python main_multiwoz.py --target_domain="Find_Attraction" --run_mode="test"
python main_multiwoz.py --target_domain="Find_Hotel"
python main_multiwoz.py --target_domain="Find_Hotel" --run_mode="test"
python main_multiwoz.py --target_domain="Find_Restaurant"
python main_multiwoz.py --target_domain="Find_Restaurant" --run_mode="test"
python main_multiwoz.py --target_domain="Find_Taxi"
python main_multiwoz.py --target_domain="Find_Taxi" --run_mode="test"
python main_multiwoz.py --target_domain="Find_Train"
python main_multiwoz.py --target_domain="Find_Train" --run_mode="test"


