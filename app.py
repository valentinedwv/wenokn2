
import json
import streamlit as st
import pandas as pd

from Kepler import my_component

st.subheader("Kepler Bidirectional Connection Demo")

map_config = my_component("""
   [ 
      { 
         info: {label: 'Bart Stops Geo', id: 'bart-stops-geo'}, 
         data: { test: 123 }
      }
   ]
""")

if map_config:
   st.code(json.dumps(json.loads(map_config), indent=4))
