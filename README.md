# Product-Retrieval-System

A system made using various techniques for effectively searching products from already provided data

**STREAMLIT APP LINK :** [link](https://appuct-retrieval-system-gzp5xbbzwqb67pqdcmbbaf.streamlit.app/)

## Steps to Reproduce 

* Use python version 3.10 (preferably)

* Clone the repository and move terminal into repo folder

1. Create Virtual environment
   
     `python -m venv [env name]`
     > 
     Activate it
   
     `[env name]\Scripts\activate`

3. Install dependencies

   `pip install -r requirements.txt`

4. Open terminal and run following command

   `streamlit run app.py`


## SOME INFO ABOUT VARIOUS FILES

* app.py is the main script file in which streamlit app is built and various other features are imported
  
* generate_embeddings.py script file is for generate various embeddings, currently they are already generated and uploaded here
  
* multilingual_query.py contains script to handle query in other standard languages around the globe
  
* model folder contains various python scripts to implement search functions for various types of models
  
* EMBEDDINGS folder contains various embeddings script files and embeddings files
  
* notebook sections contains some notebooks including EDA file
