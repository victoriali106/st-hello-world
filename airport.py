import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from matplotlib import pyplot as plt

DATAFILE = "data/airport-codes_csv.csv"


def load_data(filename):
    df = pd.read_csv(filename, na_filter=False)
    location = df["coordinates"].str.split(",")
    df["lon"] = location.apply(lambda x: float(x[0]))
    df["lat"] = location.apply(lambda x: float(x[1]))
    return df


original_data = load_data(DATAFILE)


def filter_by(data, filter_option: dict):
    filters = {}
    with st.expander("Selection Filters"):
        columns = st.columns(len(filter_option.keys()))
        for index, item in enumerate(filter_option.items()):
            key, label = item
            with columns[index]:
                unique_values = original_data[key].unique()
                options = st.multiselect(label, unique_values, [])
                filters[key] = options
    for filter, value in filters.items():
        data = data[data[filter].isin(value)]
    return data


def filter_by_range(data, filter_option: dict):
    with st.expander("Slider Filters"):
        columns = st.columns(len(filter_option.keys()))
        for index, item in enumerate(filter_option.items()):
            key, label = item
            with columns[index]:
                max_value = float(original_data[key].max())
                min_value = float(original_data[key].min())
                options = st.slider(label, min_value-1, max_value+1, (min_value, max_value))
                min_value, max_value = options
                data = data[data[key] >= min_value]
                data = data[data[key] <= max_value]
    return data


def search_by_keyword(data, key, label):
    options = st.text_input(label, "")
    if options != "":
        data = data[data[key].str.contains(options)]
    return data


def init():
    st.set_page_config(layout="wide")


ENTRY_DATA = [
    ("Main", "Introduction"),
    ("Airport type Reports", "Airport type Reports"),
    ("Airport Map", "Airport Map"),
]


def entry():
    for title, content in ENTRY_DATA:
        with st.container():
            st.header(title)
            st.markdown(f":red[{content}]")
    st.image("data/airport-photo.jpg")


def make_pie_plot(data, title, key, explode=False):
    plot, axis = plt.subplots()
    axis.set_title(title)

    labels = data[key].unique()
    if len(labels) == 0:
        return plot
    values = [data[data[key] == label].size for label in labels]
    all = sum(values)
    values = [x / all * 100 for x in values]
    if explode:
        explode = [0.1 for _ in labels]
        explode[values.index(max(values))] = 0.2
        axis.pie(values, labels=labels, explode=explode, autopct="%.2f%%")
    else:
        axis.pie(values, labels=labels, autopct="%.2f%%")
    return plot


def make_bar_plot(data, title, key):
    plot, axis = plt.subplots()
    axis.set_title(title)
    labels = data[key].unique()
    if len(labels) == 0:
        return plot
    values = [data[data[key] == label].size for label in labels]
    axis.bar(labels, values)
    plt.xticks(rotation=45)
    return plot


def make_histo_plot(data, title, key, xlabel, ylabel, horizontal=False):
    plot, axis = plt.subplots()
    axis.set_title(title)
    if len(data[key]) == 0:
        return plot

    max_value = int(data[key].max())
    min_value = int(data[key].min())
    step = max((max_value - min_value) // 10, 1)
    labels = [i for i in range(min_value, max_value + 2, step)]
    if horizontal:
        axis.histh(data[key].to_list(), labels)
        axis.set_xlabel(ylabel)
        axis.set_ylabel(xlabel)
        axis.set_xticks(labels)
    else:
        axis.hist(data[key].to_list(), labels)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(labels)
    return plot


def airport_types():
    data = filter_by(load_data(DATAFILE), {"continent": "please select a continent"})
    explode = st.checkbox(value=True, label="Explode")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(make_pie_plot(data, "Airport Types", "type", explode=explode))
    with col2:
        st.pyplot(make_bar_plot(data, "Airport Types", "type"))


def airport_map():
    data = filter_by_range(load_data(DATAFILE), {"lon": "filter by longitude",
                                                 "lat": "filter by latitude"})
    data = search_by_keyword(data, "name", "search by airport name")
    longitude = st.slider("target longitude",
                          float(data["lon"].min())-1, float(data["lon"].max())+1,
                          float(data["lon"].mean()))
    latitude = st.slider("target latitude",
                         float(data["lat"].min())-1, float(data["lat"].max())+1,
                         float(data["lat"].mean()))

    data = data[["ident", "name", "lon", "lat"]]
    data["distance"] = np.sqrt((data["lon"] - longitude) ** 2 + (data["lat"] - latitude) ** 2)
    ascending = st.selectbox("Sorting", ["Ascending", "Descending"])
    data = data.sort_values(by="distance", ascending=ascending == "Ascending")
    data = data.set_index("ident")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(data)

    # a scatterplot layer
    layer1 = pdk.Layer('ScatterplotLayer',
                       data=data,
                       get_position=['lon', 'lat'],
                       get_radius=128,
                       pickable=True)

    tool_tip = {"html": "Name: <b>{name}</b><br/> longitude: {lon}<br/>latitude: {lat}"}

    # https://deckgl.readthedocs.io/en/latest/deck.html
    if data.size == 0:
        state = pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=11)
    else:
        state = pdk.ViewState(
            latitude=data['lat'][0],
            longitude=data['lon'][0],
            zoom=11)
    map = pdk.Deck(
        map_style='light',
        layers=[layer1],
        initial_view_state=state,
        tooltip=tool_tip
    )
    with col2:
        st.pydeck_chart(map)

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(make_histo_plot(data, "Distance Distribution", "distance", "Distance", "Count"))


VIEWS = {"Main": entry,
         "Airport type Reports": airport_types,
         "Airport Map": airport_map,
         }


def create_streamlit_view():
    st.sidebar.header("Airport Codes")
    view_name = st.sidebar.radio("Menu", VIEWS.keys())
    VIEWS[view_name]()


def main():
    init()
    create_streamlit_view()


main()
