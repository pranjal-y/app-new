import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import requests
import re
from bs4 import BeautifulSoup




# Load the dataset
data = pd.read_csv('procedure_prices.csv')

# Preprocess 'procedure_price' column
data['procedure_price'] = data['procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)
data['cred_procedure_price'] = data['cred_procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)

filtered_data = data.dropna(subset=['procedure_price', 'cred_procedure_price'])


# Set the page title
st.markdown('<h1 style="text-align: center;">Medical Procedure Costs Visualization</h1>', unsafe_allow_html=True)


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


# Histogram of procedure prices with tooltips using Altair
st.markdown('<div class="white-box">', unsafe_allow_html=True)
st.write('## Histogram: Procedure Price Distribution')
if not data.empty:
    hist = alt.Chart(data).mark_bar().encode(
        x=alt.X('procedure_price:Q', bin=alt.Bin(maxbins=20), title='Procedure Price (INR)'),
        y=alt.Y('count():Q', title='Frequency'),
        tooltip=['procedure_price:Q', 'count():Q']
    ).properties(
        width=600,
        height=400,
        title='Distribution of Procedure Prices'
    )
    st.altair_chart(hist, use_container_width=True)
else:
    st.write('No valid data available for the histogram.')
st.markdown('</div>', unsafe_allow_html=True)

# Density plot of procedure prices using Altair
st.markdown('<div class="white-box">', unsafe_allow_html=True)
st.write('## Density Plot: Procedure Price Distribution')
density_chart = alt.Chart(data).mark_area().encode(
    alt.X('procedure_price:Q', title='Procedure Price (INR)'),
    alt.Y('density:Q', title='Density'),
    alt.Tooltip(['procedure_price:Q', 'density:Q'])
).transform_density(
    'procedure_price',
    as_=['procedure_price', 'density']
).properties(
    width=600,
    height=400,
    title='Density Plot of Procedure Prices'
)
st.altair_chart(density_chart, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)



# Create a double bar chart comparing procedure_price and cred_procedure_price for available data
st.write('## Double Bar Chart: Comparison of Procedure Prices')
if not filtered_data.empty:
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('Name_of_disease:N', title='Disease'),
        y=alt.Y('price:Q', title='Price (INR)', scale=alt.Scale(domain=(0, 7000000))),
        color=alt.Color('type_of_procedure:N', title='Type of Procedure', scale=alt.Scale(range=['blue', 'orange'])),
        tooltip=['Name_of_disease:N', 'price:Q', 'type_of_procedure:N']
    ).transform_fold(
        ['procedure_price', 'cred_procedure_price'],
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
available_data = data[data['procedure_price'].notnull()]
unavailable_data = data[data['procedure_price'].isnull()]

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
data1 = pd.read_csv('procedure_prices.csv')

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



























