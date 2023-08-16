import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import requests
import re
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET
import os


# Function to flatten nested JSON data
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


# Function to convert XML to flattened JSON
def xml_to_flattened_json(xml_string):
    root = ET.fromstring(xml_string)
    data = {}

    def parse_element(element, parent_key=''):
        if len(element) > 0:
            for child in element:
                parse_element(child, parent_key + element.tag + '_')
        else:
            data[parent_key + element.tag] = element.text

    parse_element(root)
    return data


# Set the page title
st.markdown('<h1 style="text-align: center;">Medical Procedure Costs Visualization</h1>', unsafe_allow_html=True)

# Load the dataset
data = pd.read_csv('procedure_prices.csv')
data1 = pd.read_csv('procedure_prices.csv')
file_path = 'procedure_prices.csv'
data_format = None

#1. Pre-processing data by converting data of json or xml to csv for further visulization.

# Check if the file format is JSON or XML
with open(file_path, 'r') as f:
    first_line = f.readline().strip()
    if first_line.startswith('{'):
        data_format = 'json'
    elif first_line.startswith('<?xml'):
        data_format = 'xml'

# Process the data based on the format
if data_format == 'json':
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    flattened_data = flatten_json(json_data)
elif data_format == 'xml':
    with open(file_path, 'r') as f:
        xml_data = f.read()
    flattened_data = xml_to_flattened_json(xml_data)
else:
    # If the format is not JSON or XML, assume it's already in CSV format
    flattened_data = None

# If flattened_data is not None, it contains the flattened data
# You can then proceed with writing it to a CSV file
if flattened_data:
    csv_file_path = 'flattened_data.csv'
    df = pd.DataFrame([flattened_data])
    df.to_csv(csv_file_path, index=False)

    st.write('Data received : Data has been flattened and saved to CSV:', csv_file_path)
else:
    st.write('Data received :Data is already in CSV format. Proceed with the next steps.')

#2. Pre-processing : Extract only the relevant columns
#Preprocess 'procedure_price' column
data['procedure_price'] = data['procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)
data['cred_procedure_price'] = data['cred_procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)
filtered_data = data.dropna(subset=['procedure_price', 'cred_procedure_price'])

# 3. Pre-processing : Deal with missing values
# Check if available data is at least 80%
available_data_percentage = data['procedure_price'].notnull().sum() / len(data)
if available_data_percentage >= 0.8:
    # Preprocess 'procedure_price' column
    data['procedure_price'] = data['procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)
    data['cred_procedure_price'] = data['cred_procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)
    filtered_data = data.dropna(subset=['procedure_price', 'cred_procedure_price'])
else:
    # Fill NA values with 0
    #data['procedure_price'].fillna(0, inplace=True)
    #data['cred_procedure_price'].fillna(0, inplace=True)
    filtered_data = data



# Create columns for layout
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="white-box">', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 35px;">Average Procedure Price</h2>', unsafe_allow_html=True)
    avg_procedure_cost = data['procedure_price'].mean()
    st.write(f"<p style='font-size: 24px;'>{avg_procedure_cost:.2f} INR</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Display Number of Procedures
with col2:
    st.markdown('<div class="white-box">', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 35px;">Number of Procedures</h2>', unsafe_allow_html=True)
    st.write(f"<p style='font-size: 24px;'>{data.shape[0]}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Display User Engagement details
with col3:
    st.markdown('<div class="white-box">', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 35px;">User Engagement</h2>', unsafe_allow_html=True)
    st.write("<p style='font-size: 24px;'>Number of visits: 5  |    Average time spent: 15    |    Most frequently accessed visualizations: pie chart</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)




# Display the dataset
st.markdown('<div class="white-box">', unsafe_allow_html=True)
st.write('## Dataset')
data_with_serial_number = data.reset_index(drop=True)  # Reset index to start from 0
data_with_serial_number.index += 1  # Start index from 1
st.write(data_with_serial_number)
st.markdown('</div>', unsafe_allow_html=True)

#Histogram - Test

# Histogram of selected column with tooltips using Altair
st.markdown('<div class="white-box">', unsafe_allow_html=True)
st.write(f'## Histogram')
# Get a list of available columns for radio buttons
available_columns = data.columns

# Create a radio button to select the column for X-axis
selected_column = st.radio("Select X-axis Column:", available_columns)

if not data.empty:
    hist = alt.Chart(data).mark_bar().encode(
        x=alt.X(f'{selected_column}:Q', bin=alt.Bin(maxbins=20), title=f'{selected_column}'),
        y=alt.Y('count():Q', title='Frequency'),
        tooltip=[f'{selected_column}:Q', 'count():Q']
    ).properties(
        width=600,
        height=400,
        title=f'Distribution of {selected_column}'
    )
    st.altair_chart(hist, use_container_width=True)
else:
    st.write(f'No valid data available for the histogram of {selected_column}.')
st.markdown('</div>', unsafe_allow_html=True)



# Density plot of procedure prices using Altair
st.markdown('<div class="white-box">', unsafe_allow_html=True)
st.write('## Density Plot')

# Get a list of available columns for X-axis selection
available_columns1 = data.columns.tolist()

# Provide a unique key for the radio button
selected_column1 = st.radio("Select X-axis Column:", available_columns1, key='density_radio')

# Create the density plot based on the selected column
density_chart = alt.Chart(data).mark_area().encode(
    alt.X(f'{selected_column1}:Q', title=selected_column1),  # Use the selected column here
    alt.Y('density:Q', title='Density'),
    alt.Tooltip([f'{selected_column1}:Q', 'density:Q'])
).transform_density(
    selected_column1,  # Use the selected column here
    as_=[selected_column1, 'density']
).properties(
    width=600,
    height=400,
    title=f'Density Plot of {selected_column1}'
)
st.altair_chart(density_chart, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


#double bar chart test
# Create a double bar chart comparing procedure_price and cred_procedure_price for available data
st.write('## Double Bar Chart: Comparison of Procedure Prices')

# Get a list of available columns for X and Y axis selection
available_columns = data.columns.tolist()

# Create dropdowns for selecting X and Y axes
selected_x_column = st.selectbox("Select X-axis Column:", available_columns)
selected_y_column = st.selectbox("Select Y-axis Column:", available_columns)

if not filtered_data.empty:
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X(f'{selected_x_column}:N', title='Disease'),
        y=alt.Y(f'{selected_y_column}:Q', title='Price (INR)', scale=alt.Scale(domain=(0, 7000000))),
        color=alt.Color('type_of_procedure:N', title='Type of Procedure', scale=alt.Scale(range=['blue', 'orange'])),
        tooltip=[f'{selected_x_column}:N', f'{selected_y_column}:Q', 'type_of_procedure:N']
    ).transform_fold(
        [selected_x_column, selected_y_column],
        as_=['type_of_procedure', 'price']
    ).properties(
        width=600,
        height=400,
        title='Comparison of Procedure Prices by Disease'
    )
    st.markdown('<div class="white-box">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.write('No valid data available for comparison.')


# Create a pie chart with hovering feature
fig = px.pie(data, values='procedure_price', names='Name_of_disease',
             title=f'Average Procedure Cost: {avg_procedure_cost:.2f} INR',
             hover_data=['Name_of_disease', 'procedure_price'],
             labels={'procedure_price': 'Procedure Cost (INR)'})

# Display the pie chart
st.write('## Pie Chart: Average Procedure Cost')
st.markdown('<div class="white-box">', unsafe_allow_html=True)
st.plotly_chart(fig)
st.markdown('</div>', unsafe_allow_html=True)

sorted_data = data.sort_values(by='procedure_price', ascending=False)

# Get the top 5 most expensive procedures
top_5_expensive = sorted_data.head(5)

# Calculate the total cost of these procedures
total_cost_top_5 = top_5_expensive['procedure_price'].sum()

# Display the table of top 5 expensive procedures
st.write('## Top 5 Most Expensive Procedures')
st.table(top_5_expensive[['Name_of_disease', 'procedure_price']])

# Create a pie chart to visualize the cost contribution
fig = px.pie(top_5_expensive, values='procedure_price', names='Name_of_disease',
             title='Cost Contribution of Top 5 Most Expensive Procedures')
st.plotly_chart(fig)

# Separate available and unavailable data
available_data = data1[data1['procedure_price'].notnull()]
unavailable_data = data1[data1['procedure_price'].isnull()]

# Calculate the count of available and unavailable data
available_count = available_data.shape[0]
unavailable_count = unavailable_data.shape[0]


# Create columns for layout
col1, col2 = st.columns(2)

# Display the available vs unavailable data using a bar chart
with col1:
    st.write('## Available vs Unavailable Data')
    # Display count of available and unavailable data
    st.write(f"Available Data Count: {available_count}")
    st.write(f"Unavailable Data Count: {unavailable_count}")
    chart = alt.Chart(pd.DataFrame({'Status': ['Available', 'Unavailable'], 'Count': [len(available_data), len(unavailable_data)]})).mark_bar().encode(
        x='Status:N',
        y='Count:Q',
        color=alt.Color('Status:N', scale=alt.Scale(range=['green', 'red'])),
        tooltip=['Status:N', 'Count:Q']
    )
    st.altair_chart(chart, use_container_width=True)

# Create a doughnut chart using Plotly
with col2:
    labels = ['Available', 'Unavailable']
    values = [available_count, unavailable_count]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    st.plotly_chart(fig, use_container_width=True)


# Define CSS styling for the buttons
button_style = """
<style>
.link-button {
    display: inline-block;
    padding: 6px 10px;
    margin: 5px;
    font-size: 12px;
    text-align: center;
    text-decoration: none;
    border: 1px solid #3498db;
    border-radius: 4px;
    cursor: pointer;
    color: #3498db;
    background-color: black;
}
.link-button:hover {
    background-color: #000080;
    color: white;
}
</style>
"""

# Apply the CSS styling
st.markdown(button_style, unsafe_allow_html=True)

#Test


# Streamlit app title
st.write('## Medical Procedure Costs Around the World')

# Load the world map shapefile
fig = px.choropleth(locations=[0], locationmode="geojson-id")


# Load the world map shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a Plotly choropleth map for interactive country selection
fig = px.choropleth(
    world,
    geojson=world.geometry,
    locations=world.index,
    color=world.index,
    hover_name="name",
    color_continuous_scale="Viridis",
    labels={'color': 'Cost'},
    title="Hover over a country to see its name and click to get cost information",
    template="plotly"
)


fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white")
fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

# Display the map and get the selected country
click_event = st.plotly_chart(fig, use_container_width=True)

if click_event:
    sorted_country_names = sorted(world["name"])
    selected_country_name = click_event.selectbox("Select a country:", sorted_country_names)

    # Get user's selection of procedure using a dropdown
    selected_procedure = st.selectbox("Select a medical procedure:", [
        "Kidney Dialysis", "Kidney Transplant", "Laparoscopic Cholecystectomy",
        "Coronary Angiography", "Normal Delivery", "Knee Replacement",
        "Brain Tumor Surgery"])

    # Construct the search query
    search_query = f"cost of {selected_procedure} in {selected_country_name}"

    # Configure serpapi request
    params = {
        "q": search_query,
        "api_key": "99a3ac874ad6a9fc7c4747826c952131625af02f56aefc64e046cb9de41dbd65"  # Replace with your actual API key
    }

    # Send request to serpapi
    response = requests.get("https://serpapi.com/search.json", params=params)
    data = response.json()

    # Extract cost information from serpapi response
    cost = None
    if "organic_results" in data:
        for result in data["organic_results"]:
            if "snippet" in result and "$" in result["snippet"]:
                cost = result["snippet"]
                break

    # Extracted cost information
    if cost:
        st.write("Estimated Cost:", cost)
        st.write("Info: The following costs are what you can roughly expect to pay for dialysis in various situations:",
                 cost)
    else:
        st.write("No cost information available for the selected combination.")

# Create a large world map with hover information using Plotly
# Load world map data
fig = px.choropleth(locations=[0], locationmode="geojson-id")

# Customize map appearance
fig.update_geos(
    showcoastlines=True,
    coastlinecolor="white",
    showland=True,
    landcolor="black",
    showocean=True,
    oceancolor="black",
    showframe=True,
    showcountries=True,
)

# Update layout
fig.update_layout(
    geo=dict(
        showframe=True,
        showcoastlines=True,
        projection_type="equirectangular",
        bgcolor="black",
    ),
    margin=dict(t=0, b=0, l=0, r=0),
    paper_bgcolor="black",
)

# Display the map with hovering feature for country names
st.plotly_chart(fig, use_container_width=True)







# Links to procedures
st.write('## Links to Procedures:')
# Create a grid layout for the buttons
num_columns = 5
num_rows = (len(data1) + num_columns - 1) // num_columns

for i in range(num_rows):
    button_row = st.columns(num_columns)
    for j in range(num_columns):
        index = i * num_columns + j
        if index < len(data1):
            with button_row[j]:
                st.markdown(f"<a class='link-button' href='{data1.iloc[index]['url']}' target='_blank'>{data1.iloc[index]['Name_of_disease']}</a>", unsafe_allow_html=True)



























