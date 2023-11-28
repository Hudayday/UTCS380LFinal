#!/bin/bash

python test_script.py GPT-Neo 0 &> GPT-Neo_0.txt
python test_script.py GPT-Neo 1 &> GPT-Neo_1.txt
python test_script.py GPT-Neo 2 &> GPT-Neo_2.txt
python test_script.py GPT-Neo 3 &> GPT-Neo_3.txt

python test_script.py GPT-J 0 &> GPT-J_0.txt
python test_script.py GPT-J 1 &> GPT-J_1.txt
python test_script.py GPT-J 2 &> GPT-J_2.txt
python test_script.py GPT-J 3 &> GPT-J_3.txt

python test_script.py GPT-NeoX 0 &> GPT-NeoX_0.txt
python test_script.py GPT-NeoX 1 &> GPT-NeoX_1.txt
python test_script.py GPT-NeoX 2 &> GPT-NeoX_2.txt
python test_script.py GPT-NeoX 3 &> GPT-NeoX_3.txt