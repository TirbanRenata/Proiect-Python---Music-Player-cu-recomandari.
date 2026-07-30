"""
Proiect final Python - Music Player cu recomandari.
"""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).parent
FOLDER_MELODII = BASE / "melodii"
FISIER_CSV = BASE / "melodii.csv"


# ----------------------------------------------------------------------
# Lucrul cu fisierele
# ----------------------------------------------------------------------

def citeste_csv():
    df = pd.read_csv(FISIER_CSV)

    if "like" not in df.columns:
        df["like"] = pd.NA

    df["like"] = df["like"].astype("object")

    df = df.reset_index(drop=True)

    return df


def scrie_csv(df):
    df.to_csv(FISIER_CSV, index=False)


def citeste_audio(gen, fisier):
    cale = FOLDER_MELODII / gen / f"{fisier}.txt"
    continut = cale.read_text()
    return base64.b64decode(continut)


def stare_like(valoare):
    if pd.isna(valoare):
        return None

    return str(valoare).strip().lower() == "true"


# ----------------------------------------------------------------------
# Modelul
# ----------------------------------------------------------------------

def pregateste_caracteristici(df, coloane=None):
    X = pd.get_dummies(df[["gen", "gen_artist"]])
    X["an"] = df["an"].astype(float)
    X["explicit"] = df["explicit"].astype(int)

    if coloane is not None:
        X = X.reindex(columns=coloane, fill_value=0)

    return X


def antreneaza_model(df):
    X = pregateste_caracteristici(df)
    y = df["like"].astype(int)
    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns


# Coeficientul de încredere NU este o probabilitate.
# El arată doar cât de ferm este răspunsul modelului.
# Un model poate avea încredere mare și totuși să greșească.
def coeficient_incredere(scor):
    scor = min(1.0, max(0.0, scor))

    return min(1.0, abs(scor - 0.5) * 2)


# ----------------------------------------------------------------------
# Interfata
# ----------------------------------------------------------------------

st.set_page_config(page_title="Music Player", layout="wide")

if "index_curent" not in st.session_state:
    st.session_state.index_curent = 0

df = citeste_csv()
genuri = sorted(df["gen"].unique())

index = int(st.session_state.index_curent) % len(df)
st.session_state.index_curent = index
melodie = df.iloc[index]

st.title("Music Player")

col_stanga, col_centru, col_dreapta = st.columns([1, 2, 1])

# --- Coloana stanga: lista de melodii ----------------------------------

with col_stanga:
    st.subheader("Playlist")
    for gen in genuri:
        with st.expander(gen):
            for i in df.index[df["gen"] == gen]:

                eticheta = df.at[i, "fisier"]

                stare = stare_like(df.at[i, "like"])

                if stare is True:
                    eticheta = "[+] " + eticheta
                elif stare is False:
                    eticheta = "[-] " + eticheta

                if st.button(eticheta, key=f"melodie_{i}"):
                    st.session_state.index_curent = int(i)
                    st.rerun()

# --- Coloana centru: playerul ------------------------------------------

with col_centru:
    st.subheader(melodie["titlu"])
    st.write("Gen:", melodie["gen"])
    st.write("An:", melodie["an"])
    st.write("Gen artist:", melodie["gen_artist"])
    st.write("Explicit:", melodie["explicit"])

    st.audio(
        citeste_audio(melodie["gen"], melodie["fisier"]),
        format="audio/mp3",
        autoplay=True
    )

    b1, b2, b3, b4 = st.columns(4)

    # LIKE
    with b1:

        if st.button("Like",use_container_width=True):

            if stare_like(df.at[index, "like"]) is True:
                df.at[index, "like"] = pd.NA
            else:
                df.at[index, "like"] = True

            scrie_csv(df)
            st.rerun()

    # DISLIKE
    with b2:
        if st.button("Dislike", use_container_width=True):
            if stare_like(df.at[index, "like"]) is False:
                df.at[index, "like"] = pd.NA
            else:
                df.at[index, "like"] = False

            scrie_csv(df)
            st.rerun()

    # SKIP
    with b3:

        if st.button("Skip", use_container_width=True):
            st.session_state.index_curent = (index + 1) % len(df)

            st.rerun()

    # Functie suplimentara- Reseteaza
    with b4:

        if st.button("Resetează toate evaluările"):
            df["like"] = pd.NA

            scrie_csv(df)

            st.session_state.index_curent = 0

            st.rerun()

    stare = stare_like(melodie["like"])

    if stare is None:
        st.caption("Neevaluată")
    elif stare:
        st.success("Ai dat Like")
    else:
        st.error("Ai dat Dislike")

# --- Coloana dreapta: predictia ----------------------------------------

with col_dreapta:
    st.subheader("Iti place?")

    evaluate = df["like"].notna().sum()

    st.progress(
        evaluate / len(df),
        text=f"{evaluate} din {len(df)} evaluate"
    )

    if evaluate != len(df):
        st.info("Panoul de predicție se deblochează după evaluarea tuturor melodiilor.")
    else:
        gen = st.selectbox("Gen", genuri)

        an = st.text_input("An")

        artist = st.radio("Gen artist", ["M", "F"])

        explicit = st.checkbox("Explicit song")

        if st.button("Prezice"):

            try:
                an = int(an)
            except ValueError:
                st.error("Anul trebuie sa fie un numar.")
                st.stop()

            model, coloane = antreneaza_model(df)

            rand = pd.DataFrame({
                "gen": [gen],
                "an": [an],
                "gen_artist": [artist],
                "explicit": [explicit]
            })

            X_nou = pregateste_caracteristici(rand, coloane)

            scor = model.predict(X_nou)[0]

            if scor > 0.5:
                st.success("Probabil îți place.")
            else:
                st.error("Probabil nu îți place.")

            st.write(f"Scor: {scor:.3f}")

            incredere = coeficient_incredere(scor)

            st.progress(incredere)

            st.write(f"Încredere: {incredere * 100:.1f}%")

            if incredere < 0.25:
                st.warning("Modelul este nesigur.")
