import pandas as pd
import streamlit as st
import seaborn as sns
import json
import joblib
import lightgbm


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data


@st.cache_data
def load_model(path):
    model = joblib.load(path)
    return model


@st.cache_data
def transform(data):
    colors = sns.color_palette("colorwarm").as_hex()
    n_colors = len(colors)

    data = data.reset_index(drop=True)

    data["label_colors"] = pd.qcut(data["price_sqm"], n_colors, labels=colors)
    data["label_colors"] = data["label_colors"].astype("str")
    return data



st.header('House Pred')

PATH_DF = "df_final.csv"
PATH_MODEL = "model_house_price.joblib"
df = load_data(PATH_DF)
model = load_model(PATH_MODEL)
st.write(df[:4])
st.write(f"df.shape: {df.shape}")
df = transform(df)
st.map(data=df, latitude='geo_lat', longitude='geo_lng')


with open(file="unique_values.json") as file:
    dict_unique = json.load(file)

# features
features = ['roomscount', 'totalarea',
       'livingarea', 'kitchenarea', 'floornumber',
       'jk_house_flat_sectionnumber', 'jk_house_flat_flatnumber',
       'jk_house_flat_flattype',
       'loggiascount', 'allroomsarea', 'geo_lat', 'geo_lng']

# for feat in features:
#         st.sidebar.slider(
#             f"{feat}", min_value=min(dict_unique[feat]), max_value=max(dict_unique[feat])
#         )

roomscount = st.sidebar.slider("roomscount", min_value=min(dict_unique["roomscount"]), max_value=max(dict_unique["roomscount"]))
totalarea = st.sidebar.slider("totalarea", min_value=min(dict_unique["totalarea"]), max_value=max(dict_unique["totalarea"]))
livingarea = st.sidebar.slider("livingarea", min_value=min(dict_unique["livingarea"]), max_value=max(dict_unique["livingarea"]))
kitchenarea = st.sidebar.slider("kitchenarea", min_value=min(dict_unique["kitchenarea"]), max_value=max(dict_unique["kitchenarea"]))
floornumber = st.sidebar.slider("floornumber", min_value=min(dict_unique["floornumber"]), max_value=max(dict_unique["floornumber"]))
jk_house_flat_sectionnumber = st.sidebar.slider("jk_house_flat_sectionnumber", min_value=1, max_value=24)
jk_house_flat_flatnumber = st.sidebar.slider("jk_house_flat_flatnumber", min_value=min(dict_unique["jk_house_flat_flatnumber"]), max_value=max(dict_unique["jk_house_flat_flatnumber"]))
loggiascount = st.sidebar.slider("loggiascount", min_value=min(dict_unique["loggiascount"]), max_value=max(dict_unique["loggiascount"]))
allroomsarea = st.sidebar.slider("allroomsarea", min_value=0, max_value=62)
geo_lat = st.sidebar.slider("geo_lat", min_value=min(dict_unique["geo_lat"]), max_value=max(dict_unique["geo_lat"]))
geo_lng = st.sidebar.slider("geo_lng", min_value=min(dict_unique["geo_lng"]), max_value=max(dict_unique["geo_lng"]))

dict_data = {"roomscount": roomscount,
             "totalarea": totalarea,
             "livingarea": livingarea,
             "kitchenarea": kitchenarea,
             "floornumber": floornumber,
             "jk_house_flat_sectionnumber": jk_house_flat_sectionnumber,
             "jk_house_flat_flatnumber": jk_house_flat_flatnumber,
             "jk_house_flat_flattype": 21,
             "loggiascount": loggiascount,
             "allroomsarea": allroomsarea,
             "geo_lat": geo_lat,
             "geo_lng": geo_lng,
             "s": 21}



data_predict = pd.DataFrame([dict_data])

button = st.button("predict")

if button:
    output = model.predict(data_predict)[0]
    st.success(f"{round(output[0])} rub / sqm")