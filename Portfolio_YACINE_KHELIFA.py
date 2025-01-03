#
import streamlit as st
from PIL import Image 
#Importation de packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import requests
from joblib import dump
import math
import os
import shap

df = pd.read_csv("Fichier_de_base.csv")
df_21 =pd.read_csv("Fichier_2021.csv")
df2 = pd.read_csv("climate_change_indicators.csv", sep=";")  


# ---------------- Menu principal ----------------
st.sidebar.title("🌟 Menu Principal")
menu = st.sidebar.radio("📌 Naviguer vers :", ["CV", "Projet", "Projet : Analyse Power BI", "Analyse Looker Studio"])

# ---------------- Section CV ----------------
if menu == "CV":
    st.title("📄 CV Interactif - Yacine KHELIFA")
    st.write("Bienvenue dans mon CV interactif. Naviguez ci-dessous pour découvrir mon parcours complet.")

    # Onglets horizontaux pour une organisation claire du CV
    tab_selected = st.selectbox(
        "🌟 Sélectionnez une section :",
        ["🏆 Lettre de Motivation", "📂 Expériences", "🛠️ Compétences", "🎓 Formations", "📞 Contact"]
    )

    # ---------------- Lettre de Motivation ----------------
    if tab_selected == "🏆 Lettre de Motivation":
        st.header("Lettre de Motivation")
        st.write("""
        **Objet** : Candidature au poste de Data Analyst

        Madame, Monsieur,  
        Passionné par l’analyse de données et diplômé d’une formation spécialisée chez Datascientest.com, 
        je maîtrise des outils comme Python, SQL et Power BI, que j’ai utilisés dans plusieurs projets 
        pour transformer des données complexes en insights exploitables.  

        Mon expérience en suivi de raccordement de la fibre optique m’a permis de développer 
        des tableaux de bord interactifs et des indicateurs clés de performance, améliorant la prise de décision 
        et la satisfaction client.  

        Rejoindre votre équipe serait une opportunité de mettre à profit mes compétences techniques et 
        humaines pour contribuer à vos projets innovants. Je suis convaincu que ma polyvalence et 
        ma capacité d’adaptation seront des atouts précieux pour votre organisation.

        Cordialement,  
        **Yacine KHELIFA**
        """)

    # ---------------- Expériences ----------------
    elif tab_selected == "📂 Expériences":
        st.header("Expériences Professionnelles")
        st.markdown("""
        ### Data Analyst - KabTelecom, Paris (Depuis 09.2024)    
        - 
        -
        -        
        ### Bénévole - Reclaim Finance, Paris (Depuis 10.2024)
        - Recherche et collecte de données sur des sites identifiés.  
        - Consolidation et traitement de données au format Excel.  

        ### Suivi de raccordement fibre optique - Orange (Depuis 06.2021)
        SERVICE ACCUEIL PRESTIGE, Paris Depuis 06.2021
        - Piloté le déploiement de fi bre optique en analysant lesdonnées de couverture et de progression dans diversesrégions.
        - Assuré le suivi des performances et le respect des délaisen utilisant des indicateurs clés de performance (KPI).
        - Mis en place des contrôles qualité basés sur l'analyse desretours clients et des données de satisfaction pour garantirune amélioration continue.
        - Créé des visualisations et des tableaux de bord interactifspour suivre les performances en temps réel et faciliter laprise de décision.
                           

        ### Conseil en gestion hôtelière - Service Accueil Prestige (01.2018 - 05.2021)
        - Dirigé une équipe conseillant l'industrie hôtelière.
        - Développé des stratégies marketing innovantes, augmentant l'occupation hôtelière.
        - Établi des partenariats pour réduire les coûts tout enmaintenant la qualité.
        - Créé des visualisations et des rapports interactifs poursuivre l'impact des stratégies marketing et optimiser lesdécisions commerciales.
                
        ### Coordonnateur services client - Hôtel Le Cheval Noir, Pantin (06.2016 - 08.2017)
        - Géré effi cacement les opérations quotidiennes del'accueil, y compris la planifi cation des quarts de travail etla coordination des tâches du personnel.
        - Mis en place des processus de suivi de la satisfactionclient et des commentaires, permettant d'identifi er lesdomaines d'amélioration.
        - Développé des tableaux de bord pour suivre lesindicateurs de satisfaction et ajuster les services enconséquence, optimisant ainsi l'effi cacité et l'expérienceclient.""")
                    
    # ---------------- Compétences ----------------
    elif tab_selected == "🛠️ Compétences":
        st.header("Compétences Techniques")
        st.markdown("""
        - **Python pour les analystes de données** : 
        Utilisation approfondie de Python pour l'extraction dedonnées, le nettoyage, l'analyse et la visualisation, entirant parti de bibliothèques telles que Pandas et NumPy. 
                     
        - **SQL pour la science des données** : 
        Techniques avancées de requêtage SQL pour manipuler etinterroger des bases de données relationnelles,essentielles pour l'extraction et la manipulation de données à grande échelle.  
                    
        - **Analyse de données** : 
        Compétences en nettoyage, transformation et analyse degrands ensembles de données pour en extraire des insights et des tendances pertinentes, utilisant des outils statistiques et algorithmiques.
                    
        - **Qualité des données** : 
        Expertise dans l'évaluation de la précision, la complétude,la validité et la cohérence des données, ainsi que dans l'implémentation de stratégies pour améliorer la qualité des données collectées ou générées.
                    
        - *Visualisation des données avec Seaborn** : 
        Capacité à créer des visualisations avancées et esthétiquement plaisantes pour interpréter les donnéesde manière intuitive, utilisant Seaborn en combinaison avec Python.
                    
        - **Matplotlib en Python** : 
        Compétences techniques pour générer des graphiques statiques, interactifs et animés avec Matplotlib, adaptés pour des analyses et des présentations de données.  
                    
        - **Statistiques exploratoires avec Python** : 
        Utilisation des méthodes statistiques exploratoires pouranalyser et interpréter des ensembles de données, permettant une compréhension approfondie des variables et des relations entre elles.

        - **Clustering avec scikit-learn** :
        Maîtrise des techniques de regroupement des données utilisant scikit-learn pour identifier des sous-ensembles significatifs et des modèles cachés dans des ensembles de données complexes.
                    

        - **Apprentissage automatique avec sklearn** :
        Développement et déploiement de modèles de machine learning, y compris la régression, la classification, et le clustering, en utilisant la bibliothèque scikit-learn.
                    
        - **Certifi cation Make (Integromat)** : 
        Compétences en automatisation des processus entre applications web via Make (anciennement Integromat), permettant de créer des flux de travail automatisés et intégrés.
                    

        - **Pipeline (FR)** : 
        Conception et gestion de pipelines de données robustes pour le traitement et l'analyse des données, en assurant l'intégrité et la disponibilité des données tout au long des processus.


        - **Power BI** : 
        Compétences avancées dans la création et la gestion de dashboards et rapports interactifs avec Power BI pour le reporting d'entreprise et l'analyse décisionnelle.
                    
       
        - **Text Mining** : 
        Techniques de fouille de texte pour l'extraction de données, l'analyse de sentiments, et la classification de documents, utilisant des outils de traitement du langage naturel.
        
        - **Web Scraping avec BeautifulSoup** : 
        Techniques pour extraire des données de sites web enutilisant BeautifulSoup, permettant de récupérer des informations structurées à partir de sources HTML ou XML.
                               


        **Je suis très motivé à l'idée de contribuer avec mon expertise en analyse de données et mes compétences en visualisation d'informations. J'accueille avec enthousiasme l'opportunité de discuter de la manière dont je pourrais apporter une valeur ajoutée. Je vous remercie pour votre considération et espère avoir l'occasion de contribuerà vos succès futurs.**

        """)

    # ---------------- Formations ----------------
    elif tab_selected == "🎓 Formations":
        st.header("Formations")
        st.markdown("""
        ### Data Analyst - DataScientest.com (01.2024 - 09.2024)
        - Formation avancée en analyse de données,
        - Maîtrise des outils statistiques et des logiciels de data science tels que Python, SQL et Power BI,
        - Développement de compétences en visualisation de données,
        - Interprétation de sets complexes,
        - Formulation de solutions basées sur les données.

        ### Licence Langue et Cultures Amazighes - Université Abderramane Mira (09.2011 - 09.2014)
        - Études approfondies sur la linguistique et la culture.  
        - Développement de compétences en analyse critique et rédaction.  

        ### Master 1 Géographie Linguistique - Université Abderramane Mira (09.2014 - 06.2015)
        - Visualisation des données : Créé des cartes pour représenter lesdonnées géographiques et linguistiques de manière claire.
        - Étude des dialectes : Analyé les variations dialectales pour comprendreles différences linguistiques à travers diverses régions.
        - Standardisation de la langue : Utilisé les données sur les variationsdialectales pour soutenir les efforts
    """)
    # ---------------- Contact ----------------
    elif tab_selected == "📞 Contact":
        st.header("Contact")
        st.write("""
        - **Téléphone** : 0641346278  
        - **Email** : [yacine.metal@hotmail.com](mailto:yacine.metal@hotmail.com)  
        - **LinkedIn** : [linkedin.com/in/yacine-khelifa](https://linkedin.com/in/yacine-khelifa)  
        - **GitHub** : [github.com/khelifaYacine](https://github.com/khelifaYacine)  
        """)

    # Lien de téléchargement du CV
        with open("Mon CV Yacine KHELIFA.pdf", "rb") as file:
          st.download_button(label="📥 Télécharger le CV (PDF)", data=file, file_name="Mon CV Yacine KHELIFA.pdf", mime="application/pdf")
    

# ---------------- Section Projet ----------------
elif menu == "Projet":
    st.title("📊 Analyse des Données du Bonheur")
    st.write("""
    Explorez les indicateurs du bonheur mondial à travers ce projet interactif.  
    Naviguez dans les étapes pour découvrir l'analyse complète.
    """)

    # Navigation verticale pour la section projet
    st.sidebar.title("Sommaire du Projet")
    pages = ["Exploration", "Data Visualisation", "Préprocessing", "Modélisation", "Interprétabilité"]
    page = st.sidebar.radio("Étapes du Projet", pages)

    if page == "Exploration":
        st.header("🔍 Exploration des Données")
        st.write("# ANALYSE DU BIEN ETRE SUR TERRE")
  # Charger une image à partir d'un fichier local
        image = "Happiness.png"
  # Afficher l'image sur l'application Streamlit
        st.image(image, caption='Les clés du bonheur', use_container_width=True)

        st.write("## Introduction")
        st.write ("Ce projet vise à conduire une analyse approfondie des données du World Happiness Report afin d'évaluer le bonheur des pays du monde en utilisant une variété d'indicateurs socio-économiques tels que la santé, la corruption, l'économie et l'espérance de vie.")
        st.write ("L'objectif principal est de présenter ces données à travers des visualisations interactives tout en identifiant les combinaisons de facteurs qui expliquent les raisons pour lesquelles certains pays sont mieux classés que d'autres en termes de bonheur.")
        st.write("")

  #importer le jeu de données du 2005 à 2020. Et 2021
  
        st.write("Affichage des 10 premières lignes du world-happiness-report.csv 2005-2020")
        st.dataframe(df.head(10))
        st.write(df.shape)

#st.write("Résumé statistique des variables numériques")
  #st.dataframe(df.describe())  
  
        st.write ("Adjonction des variables 'région' et 'température' à notre dataset ")
        def get_regional_indicator(country_name):
            country_to_region = {
                "Denmark": "Western Europe",
                "France": "Western Europe",
                "Mexico": "Latin America and Caribbean",
                "Germany": "Western Europe",
                "Poland": "Central and Eastern Europe",
                "Spain": "Western Europe",
                "Greece": "Western Europe",
                "Brazil": "Latin America and Caribbean",
                "Sweden": "Western Europe",
                "Egypt": "Middle East and North Africa",
                "Saudi Arabia": "Middle East and North Africa",
                "Lebanon": "Middle East and North Africa",
                "Netherlands": "Western Europe",
                "Australia": "Australia and New Zealand",
                "United Kingdom": "Western Europe",
                "Canada": "North America",
                "Iran": "Middle East and North Africa",
                "Pakistan": "South Asia",
                "Hungary": "Central and Eastern Europe",
                "Czech Republic": "Central and Eastern Europe",
                "Belgium": "Western Europe",
                "Turkey": "Middle East and North Africa",
                "Jordan": "Middle East and North Africa",
                "Venezuela": "Latin America and Caribbean",
                "Italy": "Western Europe",
                "Japan": "East Asia",
                "Romania": "Central and Eastern Europe",
                "Portugal": "Western Europe",
                "Singapore": "Southeast Asia",
                "Sierra Leone": "Sub-Saharan Africa",
                "Rwanda": "Sub-Saharan Africa",
                "Chile": "Latin America and Caribbean",
                "Senegal": "Sub-Saharan Africa",
                "Russia": "Commonwealth of Independent States",
                "Colombia": "Latin America and Caribbean",
                "Chad": "Sub-Saharan Africa",
                "China": "East Asia",
                "South Korea": "East Asia",
                "Slovenia": "Central and Eastern Europe",
                "Uganda": "Sub-Saharan Africa",
                "Belarus": "Commonwealth of Independent States",
                "Trinidad and Tobago": "Latin America and Caribbean",
                "Togo": "Sub-Saharan Africa",
                "Benin": "Sub-Saharan Africa",
                "Thailand": "Southeast Asia",
                "Tanzania": "Sub-Saharan Africa",
                "Bolivia": "Latin America and Caribbean",
                "Tajikistan": "Commonwealth of Independent States",
                "Taiwan Province of China": "East Asia",
                "Switzerland": "Western Europe",
                "Botswana": "Sub-Saharan Africa",
                "Sri Lanka": "South Asia",
                "Burkina Faso": "Sub-Saharan Africa",
                "Cambodia": "Southeast Asia",
                "South Africa": "Sub-Saharan Africa",
                "Cameroon": "Sub-Saharan Africa",
                "Slovakia": "Central and Eastern Europe",
                "Philippines": "Southeast Asia",
                "Costa Rica": "Latin America and Caribbean",
                "Cuba": "Latin America and Caribbean",
                "Malawi": "Sub-Saharan Africa",
                "Madagascar": "Sub-Saharan Africa",
                "Guatemala": "Latin America and Caribbean",
                "Lithuania": "Central and Eastern Europe",
                "Haiti": "Latin America and Caribbean",
                "Latvia": "Central and Eastern Europe",
                "Honduras": "Latin America and Caribbean",
                "Malaysia": "Southeast Asia",
                "Laos": "Southeast Asia",
                "Kyrgyzstan": "Commonwealth of Independent States",
                "Kuwait": "Middle East and North Africa",
                "Kenya": "Sub-Saharan Africa",
                "India": "South Asia",
                "Kazakhstan": "Commonwealth of Independent States",
                "Indonesia": "Southeast Asia",
                "Jamaica": "Latin America and Caribbean",
                "Ireland": "Western Europe",
                "Hong Kong S.A.R. of China": "East Asia",
                "Ghana": "Sub-Saharan Africa",
                "Mali": "Sub-Saharan Africa",
                "Georgia": "Commonwealth of Independent States",
                "Cyprus": "Middle East and North Africa",
                "Paraguay": "Latin America and Caribbean",
                "Panama": "Latin America and Caribbean",
                "Palestinian Territories": "Middle East and North Africa",
                "Bangladesh": "South Asia",
                "Dominican Republic": "Latin America and Caribbean",
                "Norway": "Western Europe",
                "Ecuador": "Latin America and Caribbean",
                "Nigeria": "Sub-Saharan Africa",
                "Niger": "Sub-Saharan Africa",
                "Nicaragua": "Latin America and Caribbean",
                "El Salvador": "Latin America and Caribbean",
                "New Zealand": "Australia and New Zealand",
                "Estonia": "Central and Eastern Europe",
                "Nepal": "South Asia",
                "Mozambique": "Sub-Saharan Africa",
                "Finland": "Western Europe",
                "Moldova": "Commonwealth of Independent States",
                "Peru": "Latin America and Caribbean",
                "Ukraine": "Commonwealth of Independent States",
                "Israel": "Middle East and North Africa",
                "Azerbaijan": "Commonwealth of Independent States",
                "Vietnam": "Southeast Asia",
                "Uruguay": "Latin America and Caribbean",
                "Zimbabwe": "Sub-Saharan Africa",
                "Armenia": "Commonwealth of Independent States",
                "Austria": "Western Europe",
                "Argentina": "Latin America and Caribbean",
                "United States": "North America",
                "Zambia": "Sub-Saharan Africa",
                "United Arab Emirates": "Middle East and North Africa",
                "Uzbekistan": "Commonwealth of Independent States",
                "Liberia": "Sub-Saharan Africa",
                "Bosnia and Herzegovina": "Central and Eastern Europe",
                "Montenegro": "Central and Eastern Europe",
                "Croatia": "Central and Eastern Europe",
                "Central African Republic": "Sub-Saharan Africa",
                "Mongolia": "East Asia",
                "Bulgaria": "Central and Eastern Europe",
                "Albania": "Central and Eastern Europe",
                "Mauritania": "Sub-Saharan Africa",
                "Yemen": "Middle East and North Africa",
                "Kosovo": "Central and Eastern Europe",
                "Serbia": "Central and Eastern Europe",
                "North Macedonia": "Central and Eastern Europe",
                "Belize": "Latin America and Caribbean",
                "Guyana": "Latin America and Caribbean",
                "Namibia": "Sub-Saharan Africa",
                "Afghanistan": "South Asia",
                "Djibouti": "Sub-Saharan Africa",
                "Congo (Brazzaville)": "Sub-Saharan Africa",
                "Iceland": "Western Europe",
                "Iraq": "Middle East and North Africa",
                "Syria": "Middle East and North Africa",
                "Burundi": "Sub-Saharan Africa",
                "Congo (Kinshasa)": "Sub-Saharan Africa",
                "Qatar": "Middle East and North Africa",
                "Ivory Coast": "Sub-Saharan Africa",
                "Tunisia": "Middle East and North Africa",
                "Turkmenistan": "Commonwealth of Independent States",
                "Comoros": "Sub-Saharan Africa",
                "Bahrain": "Middle East and North Africa",
                "Somaliland region": "Sub-Saharan Africa",
                "Luxembourg": "Western Europe",
                "Malta": "Western Europe",
                "Sudan": "Sub-Saharan Africa",
                "Algeria": "Middle East and North Africa",
                "Morocco": "Middle East and North Africa",
                "Swaziland": "Sub-Saharan Africa",
                "Guinea": "Sub-Saharan Africa",
                "Lesotho": "Sub-Saharan Africa",
                "Oman": "Middle East and North Africa",
                "Angola": "Sub-Saharan Africa",
                "Gabon": "Sub-Saharan Africa",
                "Mauritius": "Sub-Saharan Africa",
                "Myanmar": "Southeast Asia",
                "North Cyprus": "Western Europe",
                "Suriname": "Latin America and Caribbean",
                "Libya": "Middle East and North Africa",
                "Ethiopia": "Sub-Saharan Africa",
                "Bhutan": "South Asia",
                "Somalia": "Sub-Saharan Africa",
                "South Sudan": "Sub-Saharan Africa",
                "Gambia": "Sub-Saharan Africa",
                "Maldives": "South Asia"
            }
            if country_name in country_to_region:
                return country_to_region[country_name]
            else:
                return "Unknown"
        # Ajouter une colonne "Regional indicator" dans le DataFrame
        df["Regional indicator"] = df["Country name"].apply(get_regional_indicator)
        # Renommage des colonnes
        df_a_renamed = df2.rename(columns={'NMGB': 'Country name', 'Year': 'year'})
        # Fusion des DataFrames
        df = pd.merge(df, df_a_renamed[['Country name', 'year', 'Temperature']], on=['Country name', 'year'], how='left')
        st.dataframe(df.head())
        # Enregistrement du DataFrame dans un fichier CSV
        df.to_csv('fichier_concat.csv', index=False)

        
        if st.checkbox("Afficher les NA") :
            st.dataframe(df.isna().sum()) 








    elif page == "Data Visualisation":
        st.header("📊 Visualisation des Données")
        st.write("Graphiques interactifs pour analyser les tendances...")

        st.write ("### Distribution du score du bonheur des pays")
        fig = plt.figure()
        sns.histplot(df ["Life Ladder"], kde=True, bins=10, color="g", edgecolor="black")
        plt.title("Distribution du Life Ladder")
        plt.xlabel("Life Ladder")
        plt.ylabel("Fréquence") 
        # Calculer la médiane
        median_value = df["Life Ladder"].median()
        # Ajouter la ligne de médiane
        plt.axvline(x=median_value, color="red", linestyle="--", label="Médiane")
        # Afficher le graphique avec la légende
        plt.legend()
        st.pyplot(fig)

        #st.write ("### Interprétation :")
        st.write ("0 le moins heureux et 8 le plus heureux.")
        #st.write ("- Graphe , avec une queue plus longue vers la gauche. Cela indique qu’il y a plus de gens qui ont un score de bonheur inférieur à la moyenne que supérieur à la moyenne.")
        st.write ("")
        st.write ( "### Boxplot du score du bonheur par années")
        fig = plt.figure()
        # Sélectionner les variables catégorielles
        variables_catégorielles = [ "year"]

        # Tracer des boxplots pour chaque variable catégorielle
        for variable in variables_catégorielles:
        
            sns.boxplot(x=variable, y="Life Ladder", data=df)
            plt.title(f"Boxplot Ladder score by {variable}")
            plt.xlabel(variable)
            plt.ylabel("Ladder score")
            plt.xticks(rotation=45)
            
        st.pyplot(fig)
        
        st.write ("")
        #st.write ("### Interprétation :")
        st.write ("Légère augementation au fil des années")
        st.write ("Quelques valeurs aberrantes en 2020")

        st.write ("")
        st.write ("")
        st.write ("## Matrice de correlation par heatmap")
        #Affichage de la matrice de corrélation par heatmap
        # df_base_numeric = df.select_dtypes(include=['float64', 'int64'])
        # Calcul de la matrice de corrélation
        # corr_matrix = df_base_numeric.corr()
        # Charger une image à partir d'un fichier local
        image2 = "Heatmap.png"
        # Afficher l'image sur l'application Streamlit
        st.image(image2, use_column_width=True)
        
        st.write("")

        st.write ("### Boxenplot du score du bonheur par région")
        fig = plt.figure ()
        sns.boxenplot(x="Life Ladder", y= "Regional indicator", data = df2)
        plt.title("Distribution des scores de bonheur par région")
        plt.xlabel("Score de bonheur")
        plt.ylabel("Région")
        st.write(fig)

        st.write("")
        #st.write ("### comparaison de Life Ladder and Log GDP per capita par pays et par années")
        fig = plt.figure ()
        import plotly.express as px

        # Tracer le nuage de points interactif avec Plotly Express
        df = df.sort_values('year')
        fig = px.scatter(df,
                        x="Log GDP per capita",
                        y="Life Ladder",
                        animation_frame="year",
                        animation_group="Country name",

                        color="Social support",
                        hover_name="Country name",
                        size_max=200,
                        template="plotly_white")

        # Mettre à jour le titre du graphique
        fig.update_layout(title="Scrore du bonheur et PIB par pays selon l'année")

        # Afficher le graphique
        st.write(fig)

        st.write("")
        st.write ("### Life Ladder par pays selon l'année")
        fig = plt.figure ()
        fig = px.choropleth(df.sort_values("year"),
                        locations="Country name",
                        color="Life Ladder",
                        locationmode="country names",
                        animation_frame="year")
        fig.update_layout(title="Comparaison du Life Ladder par pays")

        st.write(fig)
        st.write ("")
        
        st.write("### Analyse du facteur climat")

        image3 = "Nuage_Temperature.png"
        # Afficher l'image sur l'application Streamlit
        st.image(image3, use_column_width=True)

    
    

    elif page == "Préprocessing":
        st.header("⚙️ Préprocessing")
        st.write("Traitement des données manquantes et préparation des variables.")

        missing = pd.read_csv ("fichier_concat.csv")    


        st.write ("# PREPROCESSING")
        # Présentation des valeurs manquantes
        st.write("## Traitement des valeurs manquantes")
        st.write("### Aperçu des valeurs manquantes dans notre dataset")
        
        # Afficher les valeurs manquantes
        st.write (missing.isnull().sum())

        # Scraping des données manquantes
        st.write("### 1. Scraping des données manquantes")
        st.write("Utilisation de l'API de la banque mondiale pour récuprer certaines valeurs manquantes de la variable 'Log GDP per capita'. ")
        #st.write("Pour la variable `Log GDP per capita`, nous avons utilisé une API pour récupérer certaines des valeurs manquantes. L'API utilisée est celle de la Banque mondiale. Cela a constitué un avantage majeur, car nous avons pu combler directement plusieurs lacunes dans notre dataset, sans devoir estimer toutes les valeurs de manière indirecte.")
        
        # Exemple de code pour scraper les données (si disponible)
        st.code("""
    import requests

    # Exemple de code pour récupérer les données manquantes
    url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json"
    response = requests.get(url)
    data = response.json()
    # Extraire les données nécessaires et les intégrer dans le dataset
        """, language='python')
        

        # Estimation des valeurs manquantes par régression linéaire
        st.write("### 2. Estimation des valeurs manquantes par régression linéaire")
        st.write("Utilisation d'estimations par régression linéaire pour les variables qui n'ont pas pu être obtenues via l'API")
        #st.write("Pour les données que nous n'avons pas pu obtenir via l'API, nous avons utilisé une méthode d'estimation par régression linéaire. Cette technique a permis de combler les valeurs manquantes en utilisant les données historiques disponibles pour chaque pays.")
        
        # Exemple de code pour la régression linéaire
        st.code("""
    # Exemple simple de régression linéaire pour estimer des valeurs manquantes
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            X = df.dropna()[['Year']]  # Par exemple, en utilisant l'année comme variable explicative
            y = df.dropna()[column]
            model = LinearRegression().fit(X, y)
            df.loc[df[column].isnull(), column] = model.predict(df.loc[df[column].isnull(), ['Year']])
        """, language='python')

        # Imputation par KNN
        st.write("### 3. Imputation par KNN")
        st.write("Lorsque la régression linéaire n'était pas suffisante nous avons appliqué une méthode d'imputation par K-nearest neighbors (KNN).")
        #st.write("Pour certaines variables où la régression linéaire n'était pas suffisante, nous avons appliqué une méthode d'imputation par K-nearest neighbors (KNN).")
        
        # Exemple de code pour l'imputation par KNN
        st.code("""
    # Imputation par KNN
    imputer = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        """, language='python')

        # Encodage des variables catégorielles
        st.write("## Encodage des variables catégorielles")
        st.write("Une fois les valeurs manquantes traitées, nous avons procédé à l'encodage des variables catégorielles afin de préparer les données pour les modèles de machine learning.")
        
        # Exemple de code pour l'encodage
        st.code("""
    # One-Hot Encoding
    encoder = OneHotEncoder()
    encoded_vars = encoder.fit_transform(df[['Country name', 'Regional indicator']])
    df_encoded = pd.concat([df.drop(columns=['Country name', 'Regional indicator']), pd.DataFrame(encoded_vars.toarray(), columns=encoder.get_feature_names_out())], axis=1)
        """, language='python')

        
            
        if st.checkbox(" ### Afficher le jeu de données après encodage") :
           final = pd.read_csv ("mon_fichier.csv")
           st.dataframe (final.isna().sum())
           st.dataframe(final)

    # Standardiastion
        st.write("## Strandardisation des variables numériques par StandardScaler")

        st.write("Ces étapes nous ont permis de finaliser notre dataset, qui est désormais prêt pour la phase de modélisation.")










    elif page == "Modélisation":
        st.header("🤖 Modélisation")
        st.write("Construction et évaluation des modèles prédictifs.")

# Scrapping de la variable "Log GDP per capita"
    ## Modelesations
    # Définir les features et la target
        df3 = pd.read_csv("mon_fichier.csv")
        X = df3.drop('Life Ladder', axis=1)  # Toutes les colonnes sauf 'Life Ladder'
        y = df3['Life Ladder']            

        # Séparer les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #pour streamlit 3
        # Définition et entraînement des modèles
        def train_model(model_type, params):
            if model_type == 'Linear Regression':
                model = LinearRegression()
            elif model_type == 'Ridge':
                model = Ridge(**params)
            elif model_type == 'Lasso':
                model = Lasso(**params)
            elif model_type == 'Elastic Net':
                model = ElasticNet(**params)
            elif model_type == 'Random Forest':
                model = RandomForestRegressor(**params)
            elif model_type == 'XGBoost':
                model = XGBRegressor(**params)

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            # Stocker le modèle et les données dans st.session_state pour la page d'interprétabilité
            st.session_state['model'] = model
            st.session_state['X_train'] = X_train

            return model, mse_train, mse_test, mae_train, mae_test, r2_train, r2_test

        # Streamlit interface
        model_type = st.selectbox('Séléctionner le type de modèle', ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'Random Forest', 'XGBoost'])
        params = {}
        if model_type in ['Ridge', 'Lasso', 'Elastic Net']:
            params['alpha'] = st.slider('Alpha', 0.01, 1.0, 0.1)
            if model_type == 'Elastic Net':
                params['l1_ratio'] = st.slider('L1 Ratio', 0.01, 1.0, 0.5)
        if model_type in ['Random Forest', 'XGBoost']:
            params['n_estimators'] = st.slider('Number of Estimators', 10, 300, 100)
            if model_type == 'XGBoost':
                params['learning_rate'] = st.slider('Learning Rate', 0.01, 0.5, 0.1)

        if st.button('Entraîner et évaluer le modèle'):
            model, mse_train, mse_test, mae_train, mae_test, r2_train, r2_test = train_model(model_type, params)
            
            # Affichage des résultats sous forme de tableau
            metrics_table = pd.DataFrame({
                'Métrique': ['MSE', 'MAE', 'R2'],
                'Entraînement': [mse_train, mae_train, r2_train],
                'Test': [mse_test, mae_test, r2_test]
            })

            st.write("### Tableau des scores des métriques")
            st.table(metrics_table)

            # Afficher les graphiques
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel('Valeurs Réelles')
            ax.set_ylabel('Prédictions')
            ax.set_title('Comparaison des Prédictions et Valeurs Réelles')
            st.pyplot(fig)







    elif page == "Interprétabilité":
        st.header("📈 Interprétabilité")
        st.write("Analyse des résultats et des variables importantes.")

    # Vérifier si le modèle a été entraîné et stocké dans st.session_state
        if 'model' not in st.session_state:
            st.write("Aucun modèle n'a été entraîné. Veuillez entraîner un modèle sur la page Modélisation.")
        else:
            model = st.session_state['model']
            X_train = st.session_state['X_train']
            X_sample = shap.sample(X_train, 100)
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

            st.subheader("Importance des caractéristiques")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            st.pyplot(fig)

            #st.subheader("Graphique en nuage de points")
            st.write("")
            st.write("#### La méthode SHAP pour le Random Forest nous permet de représenter l’importance des variables de notre jeu de données.") 

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig)

        st.write("### Interprétation des résultats")
        st.markdown("""
        - **Importance des variables** :
        - Le modèle Random Forest a identifié les variables les plus influentes sur le score de bonheur : "Log GDP per capita", "Social support", et "Healthy life expectancy at birth".
        - Les résultats soulignent l'importance des facteurs économiques et sociaux pour le bien-être des populations.
        """)

        st.write("### Conclusion")
        st.markdown("""
        - **Analyse du bien-être** :
        - L'analyse du World Happiness Report a révélé que les aspects économiques, tels que le PIB par habitant, et les facteurs sociaux, comme le support social et l'espérance de vie, sont cruciaux pour la perception du bonheur.
        - Parmi les modèles testés, le Random Forest s'est montré le plus efficace, avec un R² de 0.8919 sur les données de test, capturant bien les relations complexes entre les variables.
        """)

        st.write("")

        st.write("## Perspectives et améliorations")
        st.markdown("""
        - Explorer l'impact d'autres variables pour affiner l'étude (taux de chômage, inflation, éducation).
        - Estimer le score du bonheur pour l'année 2021 et le comparer aux données réelles.
        - Optimiser les hyperparamètres des modèles existants pour améliorer les performances de prédiction.
        - Intégrer des modèles plus sophistiqués.
        """)

        st.write("")

        st.write("## Intérêts de l'étude")
        st.markdown("""             
        - Analyse globale des facteurs influençant le bonheur mondial.
        - Base pour de futures recherches.
        - Eventuel support d'orientation de stratégies politiques.
        """)
        st.write("## Remerciements")
        st.markdown("""             
        Nous tenons à remercier Yohan et toute l'équipe de Datascientest pour leur accompagnement et leur soutien durant ces derniers mois de formation. Votre aide nous a vraiment permis de progresser et d'aller de l'avant. Un grand merci également aux membres du jury pour le temps accordé.
        """)


        with open("Rapport_Projet_Analyse_Bonheur.pdf", "rb") as file:
            st.download_button(label="📥 Télécharger le Rapport du Projet (PDF)", data=file, file_name="Rapport_Projet_Analyse_Bonheur.pdf", mime="application/pdf")


elif menu == "Projet : Analyse Power BI":
    # Titre principal
    st.title("📊 Projet : Analyse Power BI - Centre d'Appels")

    # Vidéo globale et présentation
    st.video("Enregistrement 2024-12-06 081209.mp4")  # Vidéo fournie
    st.markdown("""
    ### Présentation globale
    Ce tableau de bord Power BI analyse les performances d’un centre d’appels, avec pour objectifs :
    - Identifier les périodes de surcharge et les goulots d’étranglement.
    - Évaluer les performances des agents.
    - Proposer des axes d’amélioration pour optimiser les délais d’attente et les ressources disponibles.
    """)

    # Section 1 : Vue d'ensemble
    st.header("1️⃣ Vue d'ensemble des indicateurs clés")
    st.image("key_frame_1.png", caption="Indicateurs clés du centre d'appels")
    st.markdown("""
    **Interprétation :**
    - **Total des appels :** Le centre d'appels a traité un volume global de X appels sur une période donnée.
    - **Taux d'abandon :** Un taux significatif est observé pendant les heures de pointe, indiquant une surcharge des équipes.

    **Insights tirés :**
    - Les périodes de surcharge nécessitent une attention particulière pour éviter une augmentation des abandons.

    **Suggestions :**
    - Optimiser les plannings pour ajuster les ressources humaines pendant les pics d'activité.
    - Mettre en place un système d’automatisation pour traiter les demandes simples et réduire la charge sur les agents.
    """)

    # Section 2 : Analyse temporelle
    st.header("2️⃣ Analyse temporelle des volumes d'appels")
    st.image("key_frame_2.png", caption="Volumes d'appels par période")
    st.markdown("""
    **Interprétation :**
    - Les heures de pointe se concentrent entre 10 h - 12 h et 16 h - 18 h.
    - Une baisse notable des appels est constatée les week-ends.

    **Insights tirés :**
    - Les pics d'activité nécessitent une réallocation stratégique des ressources.
    - Les week-ends offrent une opportunité de réorganiser les tâches ou d'optimiser les temps de formation des agents.

    **Suggestions :**
    - Renforcer les équipes pendant les heures de pointe.
    - Planifier des activités annexes ou de formation pendant les périodes creuses.
    """)

    # Section 3 : Performances des agents
    st.header("3️⃣ Performances des agents")
    st.image("key_frame_3.png", caption="Comparaison des performances des agents")
    st.markdown("""
    **Interprétation :**
    - Certains agents atteignent des performances élevées, avec un faible temps de traitement par appel.
    - D'autres agents montrent une variabilité significative, avec des durées d’appel plus longues que prévu.

    **Insights tirés :**
    - Les agents les plus performants contribuent à une gestion plus rapide des appels, réduisant les délais d’attente globaux.
    - Les agents en difficulté impactent la fluidité globale du service.

    **Suggestions :**
    - Identifier les bonnes pratiques des agents performants et les partager via des formations internes.
    - Organiser des sessions de coaching ciblées pour aider les agents en difficulté à atteindre des performances optimales.
    """)

    # Section 4 : Qualité du service et délais
    st.header("4️⃣ Qualité du service et respect des délais")
    st.image("key_frame_4.png", caption="Temps moyen d'attente et SLA")
    st.markdown("""
    **Interprétation :**
    - Le temps moyen d’attente dépasse fréquemment X secondes pendant les heures de pointe.
    - Les abandons augmentent significativement après une attente prolongée de Y secondes.

    **Insights tirés :**
    - Les délais d’attente prolongés affectent directement la fluidité du service.
    - Les pics d’activité entraînent des goulots d’étranglement dans la prise en charge.

    **Suggestions :**
    - Réduire les délais d’attente en ajustant les plannings des équipes.
    - Mettre en place un système d’appel automatisé pour informer les clients de leur position dans la file.
    """)

    # Section 5 : Perspectives et améliorations
    st.header("5️⃣ Perspectives et améliorations")
    st.markdown("""
    ### Axes d'amélioration
    - **Automatisation des processus simples :** Bots ou FAQ automatisées pour traiter les demandes courantes.
    - **Prédiction des volumes d’appels :** Utiliser des modèles prédictifs basés sur les données historiques pour anticiper les besoins.
    - **Monitoring en temps réel :** Développer un tableau de bord interactif accessible aux responsables pour suivre les KPI en direct.

    ### Étapes futures
    - Formation continue des agents sur la gestion efficace des appels.
    - Optimisation des processus internes pour améliorer les performances globales.
    - Développement d’une stratégie de rétention des agents performants pour maintenir un haut niveau de qualité.
    """)



# Projet Lokker studio
elif menu == "Analyse Looker Studio":
    st.title("📊 Analyse Looker Studio : Rapport de Développement Global")
    
    # Sous-menu pour les différentes pages du projet Looker Studio
    page = st.sidebar.radio("📂 Pages du Projet :", [
        "Introduction", 
        "Population et Répartition Mondiale", 
        "Accès à l'Électricité et Internet", 
        "Progrès des Infrastructures sur 25 Ans", 
        "Synthèse et Perspectives"
    ])

    # Page Introduction
    if page == "Introduction":
        st.header("📜 Introduction Générale")
        video_path = "Vidéo sans titre (6).mp4"
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error(f"Vidéo introuvable : {video_path}")

        st.markdown("""
        ### Présentation du Projet
        Ce tableau de bord couvre **25 années de données mondiales**, offrant une vue d'ensemble des dynamiques démographiques, énergétiques et numériques :
        - **Population mondiale :** Analyse de la répartition par genre et région.
        - **Infrastructures :** Accès à l’électricité et Internet.
        - **Évolutions temporelles :** Progrès enregistrés au fil des années.

        ### Objectifs :
        - Illustrer mes compétences en **analyse de données** et **visualisation interactive**.
        - Générer des insights exploitables pour améliorer les politiques publiques et la planification.

        ⚠ **Remarque :** Certaines données sensibles sont masquées pour préserver la confidentialité.
        """)

    # Page Population et Répartition Mondiale
    elif page == "Population et Répartition Mondiale":
        st.header("🌍 Population et Répartition Mondiale")
        img_path = "Capture d'écran 2024-12-07 105210.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Population mondiale par genre et année")
        else:
            st.error(f"Image introuvable : {img_path}")

        st.markdown("""
        **Analyse :**
        - La population mondiale est estimée à **171,8 milliards**.
        - Répartition par genre :
          - **Femmes :** 85,4 milliards (50 %),
          - **Hommes :** 86,4 milliards (50 %).
        - Les régions densément peuplées incluent l'Asie (notamment la Chine et l'Inde), tandis que des zones comme l'Australie ou le Groenland ont une faible densité.

        **Insights tirés :**
        - Un équilibre démographique global entre hommes et femmes.
        - Les zones fortement peuplées nécessitent des infrastructures robustes pour soutenir la croissance (logement, santé, éducation).

        **Recommandations :**
        - **Investissements ciblés :** Prioriser les infrastructures dans les régions densément peuplées (Asie, Afrique).
        - **Planification stratégique :** Anticiper les besoins des futures générations pour éviter les pressions excessives sur les ressources naturelles et sociales.
        """)

    # Page Accès à l'Électricité et Internet
    elif page == "Accès à l'Électricité et Internet":
        st.header("🔌 Accès à l'Électricité et Internet")
        img_path = "Capture d'écran 2024-12-07 105238.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Accès global à l'électricité et Internet")
        else:
            st.error(f"Image introuvable : {img_path}")

        st.markdown("""
        **Analyse :**
        - Accès global à l’électricité : **80,6 %**.
          - **Europe :** quasi universel avec 99,99 % d’accès.
          - **Afrique :** faible accès avec seulement 43,95 %.
        - Accès global à Internet : **31,26 %**.
          - **Europe :** 56,17 % d’accès.
          - **Afrique :** seulement 11,03 %.

        **Insights tirés :**
        - Les inégalités énergétiques et numériques restent préoccupantes, impactant le développement économique et social des régions sous-développées.
        - La fracture numérique est particulièrement marquée en Afrique et dans certaines parties de l’Asie.

        **Recommandations :**
        - **Infrastructures énergétiques :** Renforcer l'accès à l'électricité en Afrique via des solutions renouvelables (solaire, éolien).
        - **Internet abordable :** Développer des politiques de soutien pour rendre Internet accessible dans les zones rurales.
        """)

    # Page Progrès des Infrastructures sur 25 Ans
    elif page == "Progrès des Infrastructures sur 25 Ans":
        st.header("📅 Progrès des Infrastructures sur 25 Ans")
        img_path = "Capture d'écran 2024-12-07 105310.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Progrès des infrastructures énergétiques et numériques")
        else:
            st.error(f"Image introuvable : {img_path}")

        st.markdown("""
        **Analyse :**
        - De 2011 à 2020, des progrès notables ont été enregistrés :
          - **Électricité :** +10 % (70 % à 80,6 %).
          - **Internet :** +21 % (10 % à 31,26 %).
        - Les régions développées (Europe, Amériques) ont enregistré des croissances stables, tandis que les progrès en Afrique et Asie restent plus lents.

        **Insights tirés :**
        - Bien que les progrès soient encourageants, les disparités régionales subsistent, impactant l’équité sociale et économique.
        - Les investissements sont inégalement répartis, favorisant principalement les régions développées.

        **Recommandations :**
        - **Initiatives internationales :** Augmenter les investissements dans les régions en développement pour accélérer la transition énergétique.
        - **Solutions locales :** Favoriser des solutions adaptées aux réalités locales (micro-grids, technologies solaires pour les zones rurales).
        """)

    # Page Synthèse et Perspectives
    elif page == "Synthèse et Perspectives":
        st.header("📜 Synthèse et Perspectives")
        st.markdown("""
        ### Synthèse des Résultats :
        - **Population mondiale :**
          - Répartition équilibrée entre hommes et femmes.
          - Zones densément peuplées nécessitant des investissements accrus dans les infrastructures de base.
        - **Accès à l'électricité et Internet :**
          - Des progrès notables, mais une fracture énergétique et numérique persistante, notamment en Afrique.
        - **Progrès des infrastructures (2011-2020) :**
          - Des améliorations constantes dans les régions développées, mais des disparités marquées subsistent dans les pays en développement.

        ### Recommandations Clés :
        - **Investissements ciblés :** Concentrer les efforts sur les zones rurales et sous-développées pour réduire les inégalités.
        - **Énergies renouvelables :** Exploiter les technologies solaires et éoliennes pour une électrification rapide des régions isolées.
        - **Internet universel :** Promouvoir des politiques pour rendre l'accès à Internet abordable et accessible dans les pays émergents.

        ### Remarque :
        Ce rapport a été conçu pour **illustrer mes compétences** en :
        - Analyse et traitement de données sur des projets complexes.
        - Création de tableaux de bord interactifs pour faciliter la prise de décision.
        - Extraction d'insights exploitables pour guider les actions stratégiques.

        ⚠ **Note Importante :** Ce tableau de bord a été réalisé avec des données limitées et simplifiées pour des raisons pédagogiques. Les analyses proposées ne doivent pas être utilisées à des fins décisionnelles sans validation complémentaire.
        """)
