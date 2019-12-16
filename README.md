


Copyright (c) 2019 The Python Packaging Authority

The package is used for risk control modeling in Python. 
It mainly provides some basic and commonly data analysis methods for partners who want to learn Python data analysis or machine learning in the Internet financial industry. 
It also contains a large number of solutions to the problems encountered by the authors in their daily work. 
I hope you can actively use it. 
If you have any questions, please contact me at hsliu_em@126.com.

### riskcontrol
Riskcontrol is used for risk control modeling in Python.
 
It provides intuitive tools for
- feature information-value (iv)
- plot bad rate of bins based on decision tree
- ks plot
- feature info describe for external data validation(contain missing analysis)
- two feature heat map for cross analysis

#### install
```
pip install riskcontrol
```

#### Usage
```python
import riskcontrol as rc
import pandas as pd

data = pd.read_csv('test.csv')

rc.feature_miss_ana(data,col_name) 

rc.split_box_plot_new(data, col_name) 

rc.ks_compute(data, col_name) 
```
