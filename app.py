## for streamlit
import streamlit as st
from streamlit_folium import folium_static


import warnings
warnings.filterwarnings("ignore")
from utils import *

st.title("COVID-19 Hospital Bed Utilization Application")
st.subheader('Main Description')
st.markdown("""
An interactive geospatial web application to help decision-makers understand the utilization of hospital bed due to COVID-19. The data is for the hospitals in the United States. This helps to effectively inform hospital planning, resource allocation and proactively mitigate the overcrowding of hospitals.    

""")

DATA_URL = ('s3://../../usa-hospital-beds.csv')

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

us_states = {
        'AL' : 'Alabama',
        'AK' : 'Alaska',
        'AS' : 'American Samoa',
        'AZ' : 'Arizona',
        'AR' : 'Arkansas',
        'CA' : 'California',
        'CO' :  'Colorado',
        'CT' : 'Connecticut',
        'DE' : 'Delaware',
        'DC' : 'District of Columbia',
        'FL' :  'Florida',
        'GA' : 'Georgia',
        'GU' : 'Guam',
        'HI' : 'Hawaii',
        'ID' : 'Idaho',
        'IL' : 'Illinois',
        'IN' : 'Indiana',
        'IA' : 'Iowa',
        'KS' : 'Kansas',
        'KY' : 'Kentucky',
        'LA' : 'Louisiana',
        'ME' : 'Maine',
        'MD' : 'Maryland',
        'MA' : 'Massachusetts',
        'MI' : 'Michigan',
        'MN' : 'Minnesota',
        'MS' : 'Mississippi',
        'MO' : 'Missouri',
        'MT' : 'Montana',
        'NE' : 'Nebraska',
        'NV' : 'Nevada',
        'NH' : 'New Hampshire',
        'NJ' : 'New Jersey',
        'NM' : 'New Mexico',
        'NY' :  'New York',
        'NC' : 'North Carolina',
        'ND' : 'North Dakota',
        'MP' : 'Northern Mariana Islands',
        'OH' :   'Ohio',
        'OK' :  'Oklahoma',
        'OR' : 'Oregon',
        'PA' : 'Pennsylvania',
        'PR' : 'Puerto Rico',
        'RI' :  'Rhode Island',
        'SC' : 'South Carolina',
        'SD' :  'South Dakota',
        'TN' : 'Tennessee',
        'TX' : 'Texas',
        'UT' : 'Utah',
        'VT' :  'Vermont',
        'VI' :  'Virgin Islands',
        'VA' :  'Virginia',
        'WA' :  'Washington',
        'WV' :  'West Virginia',
        'WI' :   'Wisconsin',
        'WY' :  'Wyoming'
}


# Load data into the dataframe.

dtf = load_data()

# Create variables
dtf['UNSTAFFED_BEDS'] = dtf['NUM_LICENSED_BEDS'] - dtf['NUM_STAFFED_BEDS']
dtf.rename({'Y':"Latitude", 'X': "Longitude"}, axis=1, inplace=True)
dtf['UTILIZATION'] = dtf.apply (lambda row: label_utilization(row), axis=1)

dtf['HQ_STATE'].replace(us_states, inplace=True)

value = st.selectbox("STATES", dtf['HQ_STATE'].unique().tolist())
#st.subheader(f"STATE {value}")
# Select State
dtf = dtf[dtf['HQ_STATE']==value][['HQ_STATE','HOSPITAL_NAME','HQ_ADDRESS','Longitude', 'Latitude', 'NUM_LICENSED_BEDS', 'NUM_ICU_BEDS', 'BED_UTILIZATION','Potential_Increase_In_Bed_Capac','UNSTAFFED_BEDS', 'UTILIZATION']].reset_index(drop=True)
dtf = dtf.reset_index().rename(columns={"index":"id"})
dtf = dtf.dropna()

location =  dtf[["Latitude", "Longitude"]][dtf['HQ_STATE']==value].head(1).values.flatten()
st.sidebar.subheader("Hospital Bed Utilization by Region Map ")
st.sidebar.markdown(

    """
    Hospital Bed Utilization by Region Map shows the utilization of beds as a measure of capacity for all adult beds and ICU beds. The maps can be viewed at the state, and drilled down to a particular hospital using the interactive geospatial plot. Low utilization means that the Total Patient Days (excluding nursery days)/Bed Days Available is less than 0.33. Medium bed utilization is between 0.33 and less than 0.66. Any hospital with more than 0.66 is categorized as high.
    """
)

st.subheader("Hospital Bed Utilization by Region Map ")
map_ = plot_map(dtf, x="Latitude", y="Longitude",start = location,  zoom=6, 
                tiles="cartodbpositron", popup="HOSPITAL_NAME", 
                size='NUM_LICENSED_BEDS', color="UTILIZATION", lst_colors=["#e6194B","#3cb44b","#4363d8"],
                marker=None)

# call to render Folium map in Streamlit
folium_static(map_)
st.image('images/utilization_legend.png', width = 75)

# Clustering
model = minisom.MiniSom(x=3, y=2, input_len=5, neighborhood_function="gaussian", activation_distance="euclidean")
model, dtf_X = fit_dl_cluster(dtf[['NUM_LICENSED_BEDS', 'NUM_ICU_BEDS', 'BED_UTILIZATION','Potential_Increase_In_Bed_Capac','UNSTAFFED_BEDS' ]], model)

# Add cluster info into original dtf
dtf[["cluster","centroids"]] = dtf_X[["cluster","centroids"]]
st.sidebar.subheader("COVID-19 Bed Utilization Clusters")
st.sidebar.markdown("""
COVID-19 Bed Utilization Cluster shows the bed utilization rate for the hospitals in the United States by clustering using using licensed beds, staffed beds, ICU beds, ventilators, , and the bed utilization rate. It helps to understand the typical bed capacity being impacted by an event ,like COVID-19.
"""
)

st.subheader("COVID-19 Bed Utilization Clusters")
# Visualize  map
map_ = plot_map(dtf, x="Latitude", y="Longitude", start=location, zoom=6, 
                tiles="cartodbpositron", popup="HOSPITAL_NAME", 
                size='NUM_LICENSED_BEDS', color="cluster", lst_colors=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'], legend=True)


folium_static(map_)
st.image('images/cluster_legend.png', width = 75)

st.sidebar.markdown("""
[Data](https://www.rearc.io/data/)
""")



st.subheader("Algorithm : Self_organizing Maps")
st.markdown(
"""
Self Organizing Map(SOM) is an unsupervised neural network machine learning technique. 
SOM is used when the dataset has a lot of attributes because it produces a low-dimensional, most of times two-dimensional, output. 
The output is a discretised representation of the input space called map.
[Reference](https://github.com/JustGlowing/minisom)
""")
st.image('images/SOM.gif', width = None)
