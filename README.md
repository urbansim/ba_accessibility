# ba_accessibility
Scripts for job accessibility analysis in Buenos Aires

To create the environment and install dependencies:
```
git clone git@github.com:urbansim/ba_accessibility.git
cd ba_accessibility
virtualenv -p python3.8 env
source env/bin/activate
pip3.8 install -r ba_accessibility/requirements.txt
```


To run the accessibility analysis:
```
cd ba_accessibility
source env/bin/activate
cd ba_accessibility
python3.8 run_analysis.py -u -st 07:00:00 -et 08:00:00 -d monday -p 1 2
```

