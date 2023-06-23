# pip install matplotlib
# pip install streamlit
# pip install scikit-learn
# pip install openpyxl
# pip install sentence_transformers
# pip install kmodes
# pip install mca

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from kmodes.kmodes import KModes
import mca
from stepmix.stepmix import StepMix

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import warnings
warnings.filterwarnings('ignore')


# What are we trying to do?
# 
# 
# Can we identify any natural groupings into personas?
# - For example men with Type II diabetes tend to be X medication and not really interested in Trials, whereas women are on y medication and really interested in Trials
# 
## What questions should we use:
# 
# From Caroline: Our questions/interests are always pretty project specific, but generally we would be most interested in observing if there is any natural clustering in our data across any of the following areas: 
# -	Area 1: Any social determinants of health (age, gender, race, ethnicity, rurality, education, income, insurance status, SES, etc.)
# -	Area 2: Burdens experienced with their condition
# -	Area 3: Typical healthcare interaction patterns/behavior (frequency, type of HCP interactions, engagement methods, etc.)
# -	Area 4: Motivations or concerns for trial enrollment
# -	Area 5: Preferences for trial engagement/involvement (communication, education, reimbursement, etc.)
# 
## A word of warning
# Keep in mind that clustering is an exploratory technique and doesn’t necessarily imply causality. It should be used together with other methods to draw robust conclusions from patient survey responses.


# First define some helper functions

def age_ranges(x):
    res = "Unknown"
    if x < 18: res = "Adolescent"
    if x >= 18 and x < 40: res = "Adult"
    if x >= 40 and x < 60: res = "Middle aged"
    if x >= 60 and x < 70: res = "Senior citizens"
    if x >= 71: res = "Elderly"
    return res


def load_data(cols=[]):
    df = pd.read_csv('survey.csv')

    # So it seems Gender was a free text field and as such is pretty noise. But every row has a value, which is good. There are some obvious fixes:
    # Male, male, M, m, Make, Male (with a space at the end), msle, Mail, Malr, maile, Mal, Cis Male, cis male, Cis Man, Male (CIS) are all 'M'
    # Female, female, F, f, Woman, Female (with a space at the end), femail, Femake, woman, Female (cis), cis-female/femme, Cis Female are all 'F'
    # For simplicity of this exercise I will group all others as 'Other'
    # NOTE: I am a strong supporter of LGBTQ+ rights and such please do not consider this simplification above as anything more for demonstration purposes for this project.
    for gender in ['Male', 'male', 'M', 'm', 'Make', 'Male ', 'msle', 'Mail', 'Malr', 'maile', 'Mal', 'Cis Male', 'cis male', 'Cis Man', 'Male (CIS)', 'Man']:
        df.loc[df['Gender'] == gender, 'Gender'] = 'Male'

    for gender in ['Female', 'female', 'F', 'f', 'Woman', 'Female ', 'femail', 'Femake', 'woman', 'Female (cis)', 'cis-female/femme', 'Cis Female']:
        df.loc[df['Gender'] == gender, 'Gender'] = 'Female'

    df['Gender'] = df['Gender'].apply(lambda x: 'Other' if x not in ['Male', 'Female'] else x)
    
    # A quick look at `comments` shows some interesting info there that could be used in a follow up experiment, but for now I will disregard the `comments` field.
    df.drop('comments', axis=1, inplace=True)

    # Let's look at 'state' first:
    # OK that largely makes sense, right, that the countries are non-US and so seemingly in this data are not using State or a State equivalent.
    # For simplicity lets fill the `nan` US ones with 'CA'.
    df.loc[(df['state'].isna()) & (df['Country'] == 'United States'), 'state'] = df['state'].mode()[0]

    # And set the rest (non-US) to 'N/A'

    df['state'] = df['state'].fillna("N/A")

    # Good, ok let's move on to `self employed`.
    # Interestingly the 18 records that do not have `self_employed` filled are the first 18 in the data_set, so maybe this was not asked fromt eh start.
    # Let's just set them to the mode of the `self_employed` column.
    df.loc[df['self_employed'].isna(), 'self_employed'] = df['self_employed'].mode()[0]

    # So finally, let's look at `work_interfere`.
    # It seems the middle of the road value of 'Sometimes' was the most answer, so let's just use that.
    df.loc[df['work_interfere'].isna(), 'work_interfere'] = df['work_interfere'].mode()[0]

    # Let's bin Age into something categorical.
    df['Age'] = df['Age'].apply(age_ranges)

    # Finally I think we can lose the `TimeStamp'.
    df.drop('Timestamp', axis=1, inplace=True)

    
    # separate continuous and categorical variable columns
    # (Although this is boilerplate I use and not really relvant as we have binned the only numerical column ('Age'))
    continuous_vars = [col for col in df.columns if df[col].dtype != 'object']
    categorical_vars = [col for col in df.columns if df[col].dtype == 'object']
    
    if len(continuous_vars) > 0:
        # Scaling is important for K-Means because K-Means is a distance-based algorithm that clusters data points based on their Euclidean distance from a centroid. If the features in the dataset are not scaled, some of them may be given higher weights than others, which can result in clustering biases towards features with larger magnitudes. This can lead to poor cluster assignments and reduced accuracy
        scaler = MinMaxScaler()
        df_con[continuous_vars] = pd.DataFrame(scaler.fit_transform(df[continuous_vars]))
    else:
        df_con = pd.DataFrame()
    
    if len(categorical_vars) > 0:
        df_cat = pd.get_dummies(df, columns=categorical_vars)
    else:
        df_cat = pd.DataFrame()
        
    df_preprocessed = pd.concat([df_con, df_cat], axis=1)
    
    return df, df_preprocessed


def do_kmeans_pca(df_preprocessed):
    # One Hot Encoding and Normalization
    preprocessor_pca = Pipeline(
        [
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    
    preprocessed_X = preprocessor_pca.fit_transform(df_preprocessed)
    
    # Select number for k
    st.write('## Identify the ideal number of clusters')
    
    # Get number of clusters via Elbow and Silhouette
    n_clusters = kmeans_elbow_silhouette(preprocessed_X)
    
    # One Hot Encoding and Normalization
    clusterer = Pipeline(
       [
           (
               "kmeans",
               KMeans(
                   n_clusters=n_clusters,
                   init="k-means++",
                   n_init=50,
                   max_iter=500,
                   random_state=42,
               ),
           ),
       ]
    )
    
    pipe = Pipeline(
        [
            ("preprocessor_pca", preprocessor_pca),
            ("clusterer", clusterer)
        ]
    )
    _ = pipe.fit(df_preprocessed)

    
    show_silhouette_score(preprocessed_X, pipe)
    
    st.write('## Show the Clusters')
    
    # Finally let's look at these as clusters

    df_pca = pd.DataFrame(
        pipe["preprocessor_pca"].transform(df_preprocessed),
        columns=["component_1", "component_2"],
    )

    # Add in the cluster
    df_pca["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_

    df_preprocessed['predicted_cluster'] = df_pca['predicted_cluster']

    show_scatter(df_pca)
 
    st.write('## Show the Important Features')
    get_important_features(df_preprocessed)

    st.write('## Show the Cluster Data')
    show_clusters(df_pca, df_preprocessed)


def do_kmeans_pca_llm(df_preprocessed):
    # Use LLM to preprocess data

    sbert_model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")

    df_embed = compile_text(df, sbert_model)

    preprocessor_pca = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )

    preprocessed_X = preprocessor_pca.fit_transform(df_embed)
    
    # Select number for k
    st.write('## Identify the ideal number of clusters')
    
    # Get number of clusters via Elbow and Silhouette
    n_clusters = kmeans_elbow_silhouette(preprocessed_X)

    clusterer = Pipeline(
       [
           (
               "kmeans",
               KMeans(
                   n_clusters=n_clusters,
                   init="k-means++",
                   n_init=50,
                   max_iter=500,
                   random_state=42,
               ),
           ),
       ]
    )
    pipe = Pipeline(
        [
            ("preprocessor_pca", preprocessor_pca),
            ("clusterer", clusterer)
        ]
    )
    _ = pipe.fit(df_preprocessed)
    
    show_silhouette_score(preprocessed_X, pipe)
    
    st.write('## Show the Clusters')
    
    # Finally let's look at these as clusters

    df_pca = pd.DataFrame(
        pipe["preprocessor_pca"].transform(df_preprocessed),
        columns=["component_1", "component_2"],
    )

    # Add in the cluster
    df_pca["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_

    df_preprocessed['predicted_cluster'] = df_pca['predicted_cluster']

    show_scatter(df_pca)
 
    st.write('## Show the Important Features')
    get_important_features(df_preprocessed)

    st.write('## Show the Cluster Data')
    show_clusters(df_pca, df_preprocessed)


def do_kmode_mca(df_preprocessed):
    n_clusters = kmode_elbow(df_preprocessed.to_numpy())
    
    kmode = KModes(n_clusters=n_clusters, init="Cao", n_init=5, verbose=0)
    clusters = kmode.fit_predict(df_preprocessed)
    
    # Do the MCA
    mca_model = mca.MCA(df_preprocessed, ncols=len(df_preprocessed.columns))
    mca_results = mca_model.fs_r(N=3)
    
    # View the eigenvalues 
    # print(mca_model.L) 

    df_preprocessed['predicted_cluster'] = clusters
    df_ca = pd.DataFrame({'component_1': mca_results[:,0], 'component_2': mca_results[:,1], 'component_3': mca_results[:,2], 'predicted_cluster': clusters})
    
    # Plot the results 
    fig = plt.figure(figsize=(8, 8))
    # Create an Axes3D object
    ax = fig.add_subplot(111, projection='3d')

    # Centroid plot 
    centroid_x = [] 
    centroid_y = []
    centroid_z = []
    for cluster in range(n_clusters): 
        xs = df_ca[df_ca['predicted_cluster'] == cluster]['component_1']
        ys = df_ca[df_ca['predicted_cluster'] == cluster]['component_2']
        zs = df_ca[df_ca['predicted_cluster'] == cluster]['component_3']
        centroid_x.append(xs.mean())
        centroid_y.append(ys.mean())
        centroid_z.append(zs.mean())

    ax.scatter(centroid_x, centroid_y, centroid_z, marker='o', color='red', s=200)
    ax.scatter(mca_results[:,0], mca_results[:,1], mca_results[:,2], c=clusters)
    ax.set_xlabel('component_1')
    ax.set_ylabel('component_2')
    ax.set_zlabel('component_3')

    # for label, x, y in zip(df_preprocessed.index, mca_results[:,0], mca_results[:,1]):
        # plt.annotate(label, xy=(x, y), xytext=(x + .02, y))
    st.pyplot(fig)
 
    st.write('## Show the Important Features')
    get_important_features(df_preprocessed)

    st.write('## Show the Cluster Data')
    show_clusters(df_ca, df_preprocessed)


def do_lca(df_preprocessed):
    # Scikit-Learn grid search object. We test n_classes from 1 to 8.
    # We also add 1-step, 2-step and 3-step to the grid.
    # We use 3 validation folds.
    # We therefore fit a total of  8x3x3=72 estimators.
    grid = {
        'n_components': [1, 2, 3, 4, 5, 6, 7, 8],
        'n_steps' : [1, 2, 3]
    }

    model = StepMix(n_components=3, n_steps=1, measurement='bernoulli', 
                    structural='gaussian_unit', random_state=42)
    
    results = dict(param_n_steps=[], param_n_components=[], aic=[], bic=[])

    # Same model and grid as above
    for g in ParameterGrid(grid):
        model.set_params(**g)
        model.fit(df_preprocessed)
        results['param_n_steps'].append(g['n_steps'])
        results['param_n_components'].append(g['n_components'])
        results['aic'].append(model.aic(df_preprocessed))
        results['bic'].append(model.bic(df_preprocessed))

    # Save results to a dataframe
    results = pd.DataFrame(results)
    
    # st.write(results)
    # st.write(results['aic'].idxmin(), results['bic'].idxmin(), results[['aic', 'bic']].idxmin(), results.loc[results[['aic', 'bic']].idxmin()])

    n_clusters = int(results.loc[results['aic'].idxmin()]['param_n_components'])
    n_steps = int(results.loc[results['aic'].idxmin()]['param_n_steps'])
    
    selected_classes = st.multiselect('Classes to show on graph', ['Class '+str(i) for i in range(n_clusters)], default=['Class '+str(i) for i in range(n_clusters)])
    
    st.write(f'### Using AIC determined best Cluster Size of {n_clusters}')
    
    # Categorical StepMix Model with n_clusters latent classes
    model = StepMix(n_components=n_clusters, n_steps=n_steps, measurement="categorical", verbose=0, progress_bar=0, random_state=123)
    # Fit model and predict clusters
    model.fit(df_preprocessed)
    clusters = model.predict(df_preprocessed)

    df_preprocessed['predicted_cluster'] = clusters
    df_ca = pd.DataFrame({'predicted_cluster': clusters})
    
    # Plot the results 
    fig = plt.figure(figsize=(8, 8))

    x_range = range(int(model.get_parameters()['measurement']['total_outcomes']  / 2))

    # for i, values in enumerate(model.get_parameters()['measurement']['pis']):
        # plt.plot(x_range, np.array(list(values)[::2]), label=f'Class {i}')

    for cls in selected_classes:
        plt.plot(x_range, np.array(list(model.get_parameters()['measurement']['pis'][int(cls[-1])][::2])), label=cls)

    # Add grid lines 
    plt.grid(True)  

    # Add a legend
    plt.legend()  

    # Add x-axis and y-axis labels
    plt.xlabel('Question and Answer')
    plt.ylabel('Question Relevancy') 

    # Add the ticks
    plt.xticks(range(len(model.feature_names_in_)), model.feature_names_in_, 
               rotation=90, fontsize=8)#, ha='right', va='bottom')
    plt.yticks([0, 1], ['Relevant', 'Not Relevant'])
               
    # Function to show plot
    st.pyplot(fig)
 
    st.write('## Show the Cluster Data')
    show_clusters(df_ca, df_preprocessed) 


def get_question_metadata():
    df, _ = load_data()
    
    question_metadata = {"unique_question_list": {}}
    for c in df.columns:
        qid = c.split(" ")[0]

        if qid == "Country":
            question_metadata["unique_question_list"][qid] = ""
        else:
            if "?" in c:
                question_metadata["unique_question_list"][qid.split("_")[0] + " " + " ".join(c.split(" ")[1:]).split("?")[0] + "?"] = ""
            else:
                question_metadata["unique_question_list"][qid.split("_")[0] + " " + " ".join(c.split(" ")[1:]).split("  ")[0] + "?"] = ""

    question_metadata["unique_question_list"] = list(question_metadata["unique_question_list"])
    question_metadata["unique_question_list_string"] = "\n".join(question_metadata["unique_question_list"])
    
    question_metadata["multi_choice_questions"] = {}

    for c in df.columns:
        qid = c.split(" ")[0]

        if qid not in ["Country", "Attention"]:
            for delim in ["?  ", ".  "]:
                if delim in c:
                    question = qid.split("_")[0] + " " + " ".join(c.split(" ")[1:]).split(delim)[0]
                    if question not in question_metadata["multi_choice_questions"]:
                        question_metadata["multi_choice_questions"][question] = {}

                        real_columns = []
                        answers = []
                        question_text = " ".join(question.split(" ")[2:])
                        for col in df.columns:
                            if question_text in col:
                                real_columns.append(col)
                                answers.append("".join((col.split(delim)[1:])))

                        question_metadata["multi_choice_questions"][question]["question_body"] = question_text
                        question_metadata["multi_choice_questions"][question]['columns'] = real_columns
                        question_metadata["multi_choice_questions"][question]['answers'] = answers
    
    return question_metadata


def kmeans_elbow_silhouette(preprocessed_X):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 1972,
    }

    # A list holds the sum of squared errors (SSE) values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(preprocessed_X)
        sse.append(kmeans.inertia_)

    # Silhouette method to select k

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    # Notice we start at 2 clusters for silhouette coefficient
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(preprocessed_X)
        score = silhouette_score(preprocessed_X, kmeans.labels_)
        silhouette_coefficients.append(score)

    fig = plt.figure(figsize=(8, 3))

    # Plot 1
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), sse)
    plt.title('Elbow method')
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")

    # Plot 2 
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.title('Silhouette method')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")

    plt.tight_layout()  # Adjust subplot padding
    st.pyplot(fig)

    # Ideal k
    n_clusters = silhouette_coefficients.index(max(silhouette_coefficients)) + 2

    st.write(f"#### Both methods indicate {n_clusters} is a good cluster size, so lets use that.")
    
    return n_clusters


def kmode_elbow(preprocessed_X):
    # For KModes, plot cost for a range of K values. Cost is the sum of all the dissimilarities between the clusters.
    # Select the K where you observe an elbow-like bend with a lesser cost value.
    # Elbow curve to find optimal K
    cost = []
    K = range(1,7)
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init="random", n_init=5)
        kmode.fit_predict(preprocessed_X)
        cost.append(kmode.cost_)
    
    fig = plt.figure(figsize=(8, 3))
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(fig)
    
    slope = np.diff(cost) / np.diff(K) 
    max_slope_idx = np.argmax(slope)
    max_slope = slope[max_slope_idx]
    threshold_slope = max_slope * 0.05

    elbow_idx = None
    for i in range(max_slope_idx, len(slope)):
        if slope[i] < threshold_slope:
            elbow_idx = i
            break
            
    n_clusters = K[elbow_idx] 
    st.write(f"#### This indicates {n_clusters} is a good cluster size, so lets use that.")
    
    return n_clusters


def show_silhouette_score(preprocessed_X, pipe):
    predicted_labels = pipe["clusterer"]["kmeans"].labels_
    ss = silhouette_score(preprocessed_X, predicted_labels)

    st.write(f"#### Silhouette Score: {round(ss, 4)} (Silhouette ranges from -1 to 1)")


def show_clusters(df_ca, df, col='predicted_cluster'):
    for cluster_number in range(len(df_ca[col].unique())):

        idx = df_ca[df_ca[col] == cluster_number].index

        st.write('### Cluster', str(cluster_number))
        st.dataframe(df.iloc[idx])


def show_scatter(df_ca):
    fig = plt.figure(figsize=(8, 8))

    scat = plt.scatter(df_ca['component_1'].to_list(), df_ca['component_2'].to_list(), c=df_ca['predicted_cluster'].to_list())

    # loop through the data and add annotations for each point
    for i in range(len(df_ca)):
        label = int(i)  # get the label from the 'index' column in the data
        x = df_ca.iloc[i]['component_1']  # get the x-coordinate
        y = df_ca.iloc[i]['component_2']  # get the y-coordinate
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title("Clustering results")
    # plt.legend(df_ca['predicted_cluster'].unique())

    st.pyplot(fig)


def get_important_features(df_preprocessed):
    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_preprocessed.drop('predicted_cluster', axis=1), df_preprocessed['predicted_cluster'])

    # Get feature importances
    importances = rf.feature_importances_

    # Sort feature importances in descending order
    sorted_indices = importances.argsort()[::-1]

    # Get sorted feature names and importances
    sorted_features = [df_preprocessed.columns[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    if len(df_preprocessed.columns) >= 5:
        # Plot feature importances as horizontal bar chart
        fig = plt.figure(figsize=(8, 8))
        plt.barh(range(5), sorted_importances[:5], align='center')
        plt.yticks(range(5), sorted_features[:5])
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        
        st.pyplot(fig)
    

def compile_text(df, sbert_model):
    res = {}
    for idx, row in df.iterrows():
        text = ""
        for col in df.columns:
            if row[col] != "": 
                text += f'{col}: "{row[col]}"\n'
        # print(text)
        # print("-----------")
        res[idx] = output_embedding(sbert_model, text.strip()).values[0]
    return pd.DataFrame(res).T


def output_embedding(sbert_model, txt):
    embd = sbert_model.encode(txt)
    return pd.DataFrame(embd.reshape(-1, 384))


# Some questions that would likely yield insightful clusters from this survey include:
# - Questions 4a-4e on awareness of different treatment types. Clustering on these could group patients by level of treatment awareness.
# - Questions 11, 12 and 13 on importance of understanding illness/treatment and actual level of understanding. Clustering on these could group patients by how well-informed they feel about their condition and care.
# - Questions 16A, 23-26 on willingness to try new/experimental treatments. Clustering on these could group patients by how open they are to novel therapies.
# - Questions 36, 38-40 on satisfaction with communication from treatment team. Clustering on these could group patients by level of satisfaction with the information provided by their doctors.
# - Questions 47-50 on motivations for and satisfaction with participating in a clinical trial. Clustering on these could provide insights into the experiences of clinical trial participants.
# 
# In general, I would recommend choosing groups of questions that relate to:
# - Awareness/understanding of illness and treatments
# - Openness to trying new therapies
# - Satisfaction with care and communication
# - Experiences with clinical trials
# 
# These types of questions are likely to yield the most meaningful clusters, representing different patient segments based on knowledge, attitudes, and experiences. The clusters could then be analyzed to determine how to best support and communicate with each patient group.

question_metadata = get_question_metadata()

# questions_area_1 = ['Q4a', 'Q4b', 'Q4c', 'Q4d', 'Q4e']
# questions_area_2 = ['Q11', 'Q12', 'Q13']
# questions_area_3 = ['Q36', 'Q37', 'Q38', 'Q39A', 'Q39B', 'Q39', 'Q40']
# questions_area_4 = ['Q47', 'Q48', 'Q49', 'Q50']

# Get list of questions 
questions = question_metadata['unique_question_list']

# Allow user to select questions
# st.sidebar.markdown('## Select questions')
selected_questions = st.sidebar.multiselect('Questions', questions, default=questions[:9] + questions[-4:])

# Allow user to select the pre-processing method
option = st.sidebar.radio('Clustering method', ['K-Means & PCA', 'K-Mode & MCA', 'Latent Class Analysis', 'K-Means & PCA & LLM'])

# if st.sidebar.button('Cluster Data'): 

selected_question_ids = [q.split(" ")[0] for q in selected_questions]
# st.write(selected_question_ids)


# Preprocess data based on option
# Create appropriate pipeline for the preprocessing.
# - Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to reduce the number of features in a dataset, while retaining as much of the original variance and information in the dataset as possible. Therefore PCA can be useful for preprocessing data before clustering

# Load the data
df, df_preprocessed = load_data(selected_question_ids)

if option == 'K-Means & PCA': do_kmeans_pca(df_preprocessed)
if option == 'K-Mode & MCA': do_kmode_mca(df_preprocessed)
if option == 'Latent Class Analysis': do_lca(df_preprocessed)
if option == 'K-Means & PCA & LLM': do_kmeans_pca_llm(df_preprocessed)
