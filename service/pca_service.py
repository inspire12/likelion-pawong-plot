import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px

import ast  # 문자열을 실제 리스트로 변환하기 위한 라이브러리

class PCAService:
    def __init__(self):
        self.embedding_index = 28
        self.scaler = StandardScaler()
        self.df = pd.read_csv('data/refined_output.csv')
        # self.df = pd.read_csv('data/embed_test_data.csv')

    def __parse_embedding(self, embedding_str):
        cleaned_str = embedding_str.strip().strip('[]')
        cleaned_str = cleaned_str.replace(";", " ")
        return np.array([float(num) for num in cleaned_str.split()])

    def visualize_2d(self):

        embeddings = self.df.iloc[:, self.embedding_index].apply(self.__parse_embedding)
        embeddings_matrix = np.vstack(embeddings.values)  # 2D 배열로 변환

        # 3. PCA로 차원 축소 (1536차원 → 2차원으로 축소)
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)

        # PCA 결과를 데이터프레임으로 만듦 (색상과 Hover 정보 포함)
        result_df = pd.DataFrame({
            'PC1': reduced_embeddings[:, 0],
            'PC2': reduced_embeddings[:, 1],
            'original_info': self.df.iloc[:, 1],  # 마우스 Hover 정보 (원하는 컬럼 지정)
            'kind': self.df.iloc[:, 6]  # 색상 기준 컬럼 (원하는 컬럼 지정)
        })

        # PCA 결과 시각화
        fig = px.scatter(
            result_df, x='PC1', y='PC2',
            color='kind',  # 색상으로 사용할 컬럼
            hover_data=['special_mark'],  # 마우스 Hover 시 보여줄 컬럼
            title='Interactive PCA Visualization (2D)'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(height=600, width=800)

        fig.show()

    def visualize_3d(self):
        # 데이터 정제

        embeddings = self.df.iloc[:, self.embedding_index].apply(self.__parse_embedding)
        embeddings_matrix = np.vstack(embeddings.values)
        # PCA 를 통해 차원 압축
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)

        # PCA 결과 DataFrame 생성 (원본 데이터 정보 포함)
        result_df = pd.DataFrame({
            'PC1': reduced_embeddings[:, 0],
            'PC2': reduced_embeddings[:, 1],
            'PC3': reduced_embeddings[:, 2],
            'kind': self.df.iloc[:, 6],  # 마우스 올렸을 때 정보 확인
            'special_mark': self.df.iloc[:, 19] # 분포 색깔 확인
        })

        # PCA가 설명하는 분산 비율 출력 (각 PC의 중요성 파악)
        explained_variance = pca.explained_variance_ratio_
        print(f'PC1이 설명하는 비율: {explained_variance[0]:.2%}')
        print(f'PC2가 설명하는 비율: {explained_variance[1]:.2%}')
        print(f'PC3가 설명하는 비율: {explained_variance[2]:.2%}')
        print(f'총 설명된 분산 비율: {explained_variance.sum():.2%}')

        # 3D Plotly로 interactive 시각화
        fig = px.scatter_3d(
            result_df, x='PC1', y='PC2', z='PC3',
            color='kind',
            hover_data=['special_mark'],
            title='3D PCA Visualization of Embeddings'
        )
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(height=700, width=800)
        fig.show()

    def print_pca_pivot_means(self, reduced_embeddings):
        # 워드 임베딩한 정보를 통해 처리하다보니 임베딩 필드의 각 차원이 의미하는바를 파악하기 어렵다
        raise NotImplemented
        # 원본 데이터의 수치형 컬럼만 뽑아옴
        #
        # original_features = self.df.select_dtypes(include=[np.number])
        #
        # # PCA 결과와 원본 데이터를 합쳐 상관관계 분석
        # pca_df = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2', 'PC3'])
        # merged_df = pd.concat([pca_df, original_features], axis=1)
        #
        # # 상관관계 계산
        # corr_matrix = merged_df.corr()
        #
        # # PCA 축별 상관관계 결과 보기 좋게 정렬
        # print("📌 PC1과 원본 데이터의 상관관계 (절대값이 큰 순서로 정렬)")
        # print(corr_matrix['PC1'].abs().sort_values(ascending=False).head(10))
        #
        # print("\n📌 PC2와 원본 데이터의 상관관계 (절대값이 큰 순서로 정렬)")
        # print(corr_matrix['PC2'].abs().sort_values(ascending=False).head(10))
        #
        # print("\n📌 PC3와 원본 데이터의 상관관계 (절대값이 큰 순서로 정렬)")
        # print(corr_matrix['PC3'].abs().sort_values(ascending=False).head(10))
