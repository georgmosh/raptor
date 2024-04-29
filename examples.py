import os
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.cluster_tree_builder import ClusterTreeConfig, ClusterSpatialConfig, ClusterHardConfig
from raptor.EmbeddingModels import SBertEmbeddingModel, OpenAILargeEmbeddingModel
from raptor.SummarizationModels import BlindSummarizationReverseModel
from raptor.QAModels import GPT3Turbo16kQAModel

os.environ["OPENAI_API_KEY"] = "I_LOVE_LASAGNA"
os.environ["CLUSTER_MAXIMUM_SIZE"] = "10"

# Cinderella story defined in sample.txt
with open('demo/sample.txt', 'r') as file:
    text = file.read()

large_embeddings = OpenAILargeEmbeddingModel()

tree_builder_config = ClusterTreeConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=None,
    embedding_models={"EMB": large_embeddings},
    cluster_embedding_model="EMB",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(
    tree_builder_config=tree_builder_config,
    embedding_model=large_embeddings
)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

tree_builder_config = ClusterSpatialConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=None,
    embedding_models=None,
    cluster_embedding_model="OpenAI",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(
    tree_builder_config=tree_builder_config
)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

tree_builder_config = ClusterSpatialConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=None,
    embedding_models={"EMB": large_embeddings},
    cluster_embedding_model="EMB",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(
    tree_builder_config=tree_builder_config,
    embedding_model=large_embeddings)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

tree_builder_config = ClusterHardConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=None,
    embedding_models=None,
    cluster_embedding_model="OpenAI",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(
    tree_builder_config=tree_builder_config
)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

tree_builder_config = ClusterHardConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=None,
    embedding_models={"EMB": large_embeddings},
    cluster_embedding_model="EMB",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(
    tree_builder_config=tree_builder_config,
    embedding_model=large_embeddings
)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

tree_builder_config = ClusterSpatialConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=BlindSummarizationReverseModel(),
    embedding_models=None,
    cluster_embedding_model="OpenAI",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(tree_builder_config=tree_builder_config,
                                                            qa_model=GPT3Turbo16kQAModel())
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

tree_builder_config = ClusterSpatialConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=BlindSummarizationReverseModel(),
    embedding_models={"EMB": large_embeddings},
    cluster_embedding_model="EMB",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(tree_builder_config=tree_builder_config,
                                                            qa_model=GPT3Turbo16kQAModel(),
                                                            embedding_model=large_embeddings)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)
