# ba_accessibility
Scripts to conduct job accessibility analysis in Buenos Aires

####To get started and run the analysis:
```
1. Install python 3.8 
2. Install pip3: sudo apt-get install python3-pip
3. Install virtualenv: sudo apt install virtualenv
4. Clone ba_accessibility repository: git clone git@github.com:urbansim/ba_accessibility.git
5. Navigate to the main directory: cd ba_accessibility
6. Create a clean python 3.8 environment: virtualenv -p python3.8 env
7. Activate the new environment: source env/bin/activate
8. Install repo requirements: pip3.8 install -r ba_accessibility/requirements.txt
9. Navigate to the internal ba_accessibility folder: cd ba_accessibility
10. Execute the run_analysis script passing appropriate parameters
```

####The parameters of the `run_analysis.py` script are:

• `ud`: Update Demographics flag. Must be passed the first time the script is executed and any time the jobs or population data changes. 

• `ug`: Update GTFS flag. Must be passed the first time the script is executed and any time the input GTFS files change. 

• `p`: List of project ids to process separated by spaces. At least one project id must be passed

• `st`: Start time for_analysis in 24 hr format (ej 07:00)

• `et`: End time for analysis in 24 hr format (ej 08:00)

• `d`: Week day for analysis in 24 hr format (ej monday)


####Example

The example below runs the initial accessibility analysis for projects 1 and 2, trips between 7 a.m. and 8 a.m. of a typical monday.
```
python3.8 run_analysis.py -ud -ug -p 1 2 -st 07:00:00 -et 08:00:00 -d monday 
```

####Outputs



####Computing Requirements
The complete analysis for two projects runs in about 6 hours in a machine with 32GB of RAM. 



