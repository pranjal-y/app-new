import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Load the dataset
data = pd.read_csv('procedure_prices.csv')

# Preprocess 'procedure_price' column
data['procedure_price'] = data['procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)
data['cred_procedure_price'] = data['cred_procedure_price'].str.replace('INR', '').str.replace(',', '').astype(float)

filtered_data = data.dropna(subset=['procedure_price', 'cred_procedure_price'])

# Set the page title
st.title('Medical Procedure Costs Visualization')

# Display the dataset
st.write('## Dataset')
st.write(data)

# Histogram of procedure prices with tooltips using Altair
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

# Density plot of procedure prices using Altair
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
    st.altair_chart(chart, use_container_width=True)
else:
    st.write('No valid data available for comparison.')

# Calculate average procedure cost
avg_procedure_cost = data['procedure_price'].mean()

# Create a pie chart with hovering feature
fig = px.pie(data, values='procedure_price', names='Name_of_disease',
             title=f'Average Procedure Cost: {avg_procedure_cost:.2f} INR',
             hover_data=['Name_of_disease', 'procedure_price'],
             labels={'procedure_price': 'Procedure Cost (INR)'})

# Display the pie chart
st.plotly_chart(fig)

# Create a line chart with hovering feature
line_chart = alt.Chart(data).mark_line().encode(
    x='Name_of_disease:N',
    y=alt.Y('value:Q', title='Procedure Price (INR)'),
    color='variable:N',
    tooltip=['Name_of_disease:N', 'value:Q', 'variable:N']
).transform_fold(
    ['procedure_price', 'cred_procedure_price'],
    as_=['variable', 'value']
).properties(
    width=800,
    height=500,
    title='Procedure Price Trends'
)
st.altair_chart(line_chart, use_container_width=True)

# Links to procedures
st.write('## Links to Procedures:')
for index, row in data.iterrows():
    st.write(f"[{row['Name_of_disease']}]({row['url']})")







