import os
import torch
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage
import matplotlib.pyplot as plt

# --- 模型加载 ---
model_path = "/home/student/zzc/deepseek/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# --- Pipeline 创建 ---
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Embedding 模型 ---
embeddings = HuggingFaceEmbeddings(
    model_name="/home/student/zzc/代码/GAI-agent-satellite-main/embedding",
    model_kwargs={'device': 'cuda:0'}
)


# --- Block and Sub-block Representations and Embeddings ---
block_representations = {
    "Scenarios": "Knowledge encompassing diverse satellite network deployment scenarios, including variations in constellation types (e.g., GEO, LEO), application-specific mission objectives (e.g., communication, Earth observation, navigation, scientific), and architectural configurations (e.g., bent-pipe, regenerative payload, inter-satellite links, ground segment topologies).",
    "Access Protocols": "Knowledge concerning various multiple access protocols employed in satellite communication systems, such as Space Division Multiple Access (SDMA) utilizing beamforming, Rate-Splitting Multiple Access (RSMA) for non-orthogonal transmission. Understanding the operational principles, performance trade-offs, and application suitability of each protocol is crucial.",
    "Channel Models": "Knowledge of different channel models relevant to satellite communication links, covering both static and dynamic channel conditions. This includes fixed channel models like Additive White Gaussian Noise (AWGN) for idealized scenarios and time-varying channel models to represent realistic impairments such as fading (e.g., Rician, Rayleigh, Nakagami-m), shadowing due to obstacles, atmospheric absorption, scintillation effects, rain attenuation, and Doppler frequency shifts caused by satellite motion. Understanding the statistical properties and parameters of these models is essential for link budget analysis and system design.",
    "Optimization Goals": "Knowledge regarding various performance optimization objectives in satellite communication system design and operation. This includes maximizing Spectral Efficiency (SE) to improve data rates, enhancing Energy Efficiency (EE) to reduce power consumption, increasing system throughput for higher capacity, minimizing communication latency for real-time applications, ensuring fairness in resource allocation among multiple users, and guaranteeing Quality of Service (QoS) requirements for different service types.  Understanding the mathematical formulations of these objectives and the trade-offs between them is important for algorithm development and resource management."
}

sub_block_representations = {
    "Scenarios": {
        "Homogeneous": "Homogeneous satellite network scenarios define constellations where all satellites possess uniform characteristics, including orbital altitude, coverage area, capabilities, and service provision. These scenarios are used for deploying and managing uniform constellations for specific applications, such as global broadband internet access via Low Earth Orbit (LEO) constellations, where consistency in service and design is crucial.",
        "Heterogeneous": "Heterogeneous satellite network scenarios define constellations composed of satellites with diverse characteristics, including mixed orbital regimes (LEO, MEO, GEO), varied payloads (communication, Earth observation), and multi-layer network architectures. These scenarios are used to achieve enhanced coverage, capacity, and service diversity by integrating different satellite types, addressing complex application requirements like multi-service platforms and integrated Earth observation and communication systems."
    },
    "Access Protocols": {
        "SDMA": "Space Division Multiple Access (SDMA) protocol spatially separates users via beamforming techniques, including fixed, steerable, and adaptive beams, to enhance frequency reuse and system capacity in satellite networks. It's used to manage interference and improve efficiency by directing signals to specific user locations.",
        "RSMA": "Rate-Splitting Multiple Access (RSMA) protocol improves spectral efficiency through non-orthogonal transmission, splitting user messages into common and private streams. Utilizing schemes like basic and enhanced RSMA, and employing successive interference cancellation (SIC) at receivers, it handles heterogeneous user demands and improves system throughput compared to orthogonal access methods in satellite communications."
    },
    "Channel Models": {
        "Fixed": "Fixed channel models, like the AWGN model, assume constant or slowly varying channel parameters (signal attenuation, noise power spectral density). These idealizations are used for initial link budget calculations, assessing performance under ideal conditions, and in static scenarios (fixed ground station to geostationary satellite), calculating path loss and antenna gain. They provide baselines for system design and are widely used in education.",
        "Time-Varying": "Time-varying channel models account for channel parameter variations over time, accurately representing actual satellite communication. These models, like Rayleigh (NLOS), Rician (LOS with multipath), shadowing (obstruction-induced signal loss), and those describing Doppler shift, are used for simulating dynamic scenarios (mobile and LEO satellite communication), optimizing link performance in complex environments, designing anti-fading techniques, and developing Doppler compensation technologies."
    },
   "Optimization Goals": {
        "SE": "Spectral Efficiency (SE) optimization in satellite communications focuses on maximizing data rate per unit bandwidth through techniques like advanced modulation and coding, signal processing, interference management, and resource allocation. It's used to enhance data throughput in limited bandwidth scenarios, improving overall link and network capacity.",
        "EE": "Energy Efficiency (EE) optimization in satellite communications aims to minimize power consumption while maintaining performance, employing methods such as power amplifier optimization, power control, energy-aware resource management, and efficient hardware design. It's crucial for extending the operational life of battery-powered terminals and payloads, especially in long-duration satellite missions."
    }
}
block_representation_embeddings = {
    block_name: embeddings.embed_query(representation)
    for block_name, representation in block_representations.items()
}

sub_block_representation_embeddings = {}
for block_name, sub_blocks in sub_block_representations.items():
    sub_block_representation_embeddings[block_name] = {
        sub_block_name: embeddings.embed_query(representation)
        for sub_block_name, representation in sub_blocks.items()
    }

# --- Routing Functions ---
def route_to_block(query_embedding, block_representation_embeddings):
    similarity_scores = {
        block_name: torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(block_embedding),
            dim=0
        ).item()
        for block_name, block_embedding in block_representation_embeddings.items()
    }
    selected_block = max(similarity_scores, key=similarity_scores.get)
    print(f"Layer-1 Routing: Selected Block - {selected_block}")
    return selected_block

def route_to_sub_block(query_embedding, sub_block_representation_embeddings, selected_block):
    similarity_scores = {
        sub_block_name: torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(sub_block_embedding),
            dim=0
        ).item()
        for sub_block_name, sub_block_embedding in sub_block_representation_embeddings[selected_block].items()
    }
    selected_sub_block = max(similarity_scores, key=similarity_scores.get)
    print(f"Layer-2 Routing: Selected Sub-block - {selected_sub_block} within {selected_block}")
    return selected_sub_block

def ask_deepseek(user_input, retrievers_dict, block_representation_embeddings, sub_block_representation_embeddings, pipe, embeddings):
    # 1. 编码用户查询
    query_embedding = embeddings.embed_query(user_input)

    # 2. Layer-1 路由 (选择 Block)
    selected_block = route_to_block(query_embedding, block_representation_embeddings)

    # 3. Layer-2 路由 (选择 Sub-block)
    selected_sub_block = route_to_sub_block(query_embedding, sub_block_representation_embeddings, selected_block)

    # 4. 基于路由结果选择 Retriever
    retriever_key = f"{selected_block}_{selected_sub_block}"
    retriever = retrievers_dict.get(retriever_key, retrievers_dict["Access Protocols_SDMA"]) # 默认使用 retriever_SDMA

    retrieved_chunks = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

    # Modified Prompt
    prompt_content = f"""You are an expert in satellite communications. Formulate a satellite communication model in English based on the following background knowledge to answer the user's problem directly.
    Background Knowledge:
    {context}

    User Problem:
    {user_input}
    """
    #generated_text = pipe(prompt_content)[0]['generated_text']
    response_content = "generated_text"

    return response_content, selected_block, selected_sub_block


def evaluate_retrieval_rate(queries_annotation, retrievers_dict, block_representation_embeddings, sub_block_representation_embeddings, pipe, embeddings):
    correct_retrieval_count = 0
    total_queries = len(queries_annotation)
    print(len(queries_annotation))

    for query, annotation in queries_annotation.items():
        _, selected_block, selected_sub_block = ask_deepseek(query, retrievers_dict, block_representation_embeddings, sub_block_representation_embeddings, pipe, embeddings)
        correct_block = annotation["block"]
        correct_sub_block = annotation["sub_block"]

        if selected_block == correct_block:
            if selected_sub_block == correct_sub_block:
                correct_retrieval_count += 1
            else:
                correct_retrieval_count += 0.2

    retrieval_rate = (correct_retrieval_count / total_queries) * 100
    return retrieval_rate


# --- 评估数据集 (示例，你需要根据实际情况创建和标注) ---
annotation_queries = {
#------简单一点的（论文里面给的）
    "My application covers a wide range and is complicated. Show me the stakeholdersand relationship in the satellite network. Answer using your local knowledge.": {"block": "Scenarios", "sub_block": "Heterogeneous"}, 
    "The satellite states and environment are changing with time, which kind of channelIshould use? Answer based on your local knowledge": {"block": "Channel Models", "sub_block": "Time-Varying"}, 
    "Since heterogeneous satellite networks are considered, to mitigate interferenceplease use the RSMA protocol at the LEO satellite.": {"block": "Access Protocols", "sub_block": "RSMA"}, 
    "I would like to maximize the system sum rate of LEO satellite area. Show me theformulated problem. Use your local knowledge": {"block": "Optimization Goals", "sub_block": "SE"}, 
    "The situation changes. Now, the wireless environment is fixed. What should l doto revise the formulated problem accordingly.": {"block": "Channel Models", "sub_block": "Fixed"}, 
#-----更复杂抽象的（gpt生成的）
    "Tell me about situations where all satellites in a network are quite similar.": {"block": "Scenarios", "sub_block": "Homogeneous"}, # 更加口语化的 Homogeneous 场景描述
    "How can we differentiate network architecture when satellites are very diverse in capability?": {"block": "Scenarios", "sub_block": "Heterogeneous"}, #  更口语化的 Heterogeneous 场景描述
    "What are the methods for allocating resources based on user location in satellite downlink?": {"block": "Access Protocols", "sub_block": "SDMA"}, #  关注用户位置资源分配，暗示 SDMA
    "Describe an approach where data for different users is split and superimposed before sending from the satellite.": {"block": "Access Protocols", "sub_block": "RSMA"}, #  描述 RSMA 数据分割和叠加的特点
    "What are the common channel conditions when the satellite's position relative to ground stations remains unchanged?": {"block": "Channel Models", "sub_block": "Fixed"}, #  描述卫星相对地面站位置不变的情况，暗示 Fixed Channel
    "Explain how the channel quality changes as the satellite orbits around the Earth.": {"block": "Channel Models", "sub_block": "Time-Varying"}, #  描述卫星轨道运动导致信道质量变化，暗示 Time-Varying Channel
    "What optimization strategies prioritize minimizing power consumption for satellite payloads?": {"block": "Optimization Goals", "sub_block": "EE"}, # 更明确关注功率消耗的优化目标，暗示 EE
    "How can we design a satellite system to maximize the data throughput for each unit of bandwidth?": {"block": "Optimization Goals", "sub_block": "SE"}, #  更明确关注频谱效率的优化目标，暗示 SE
    "Discuss satellite network configurations where all satellites are of the same type.": {"block": "Scenarios", "sub_block": "Homogeneous"}, #  另一种 Homogeneous 场景描述方式
    "Explain the challenges in designing networks with varied types of satellites.": {"block": "Scenarios", "sub_block": "Heterogeneous"}, #  另一种 Heterogeneous 场景描述方式
    "How does spatial domain multiplexing work in satellite communications to serve multiple users?": {"block": "Access Protocols", "sub_block": "SDMA"}, #  使用 “spatial domain multiplexing”  暗示 SDMA
    "Describe the concept of non-orthogonal access by splitting rates for satellite users.": {"block": "Access Protocols", "sub_block": "RSMA"}, #  使用 "non-orthogonal access by splitting rates" 描述 RSMA
    "What are the channel characteristics when considering a geostationary satellite?": {"block": "Channel Models", "sub_block": "Fixed"}, #  使用 “geostationary satellite”  暗示 Fixed Channel
    "Explain the factors that cause channel variations in satellite-to-ground links over time.": {"block": "Channel Models", "sub_block": "Time-Varying"}, #  更详细描述 Time-Varying Channel 的成因
    "What are the optimization aims when we focus on maximizing bits per second per Hertz in satellite communication?": {"block": "Optimization Goals", "sub_block": "SE"}, # 使用 “bits per second per Hertz” 描述频谱效率
    "How can satellite systems be optimized for minimal energy per bit transmitted?": {"block": "Optimization Goals", "sub_block": "EE"}, #  使用 “energy per bit transmitted” 描述能效
}


# --- 超参数调优和绘图 ---
chunk_sizes = [500, 1000] 
chunk_numbers = [1, 2]
rr_results = {} #

# ---  Retriever 字典 (创建所有子块的 Retriever) ---
retrievers_dict = {}
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
# --- 加载和处理知识库 ---
loader_Scenarios_Homogeneous = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_Scenarios_Homogeneous.txt')
documents_Scenarios_Homogeneous = loader_Scenarios_Homogeneous.load()
texts_Scenarios_Homogeneous = text_splitter.split_documents(documents_Scenarios_Homogeneous) 
vectordb_Scenarios_Homogeneous = Chroma.from_documents(texts_Scenarios_Homogeneous, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Scenarios_Homogeneous_db")
vectordb_Scenarios_Homogeneous.persist()

loader_Scenarios_Heterogeneous = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_Scenarios_Heterogeneous.txt')
documents_Scenarios_Heterogeneous = loader_Scenarios_Heterogeneous.load()
texts_Scenarios_Heterogeneous = text_splitter.split_documents(documents_Scenarios_Heterogeneous)
vectordb_Scenarios_Heterogeneous = Chroma.from_documents(texts_Scenarios_Heterogeneous, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Scenarios_Heterogeneous_db")
vectordb_Scenarios_Heterogeneous.persist()

loader_SDMA = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_SDMA.txt')
documents_SDMA = loader_SDMA.load()
texts_SDMA = text_splitter.split_documents(documents_SDMA) 
vectordb_SDMA = Chroma.from_documents(texts_SDMA, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/SDMA_db")
vectordb_SDMA.persist()

loader_RSMA = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_RSMA.txt')
documents_RSMA = loader_RSMA.load()
texts_RSMA = text_splitter.split_documents(documents_RSMA) 
vectordb_RSMA = Chroma.from_documents(texts_RSMA, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/RSMA_db")
vectordb_RSMA.persist()

loader_Channels_Fixed = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_Channels_Fixed.txt')
documents_Channels_Fixed = loader_Channels_Fixed.load()
texts_Channels_Fixed = text_splitter.split_documents(documents_Channels_Fixed)
vectordb_Channels_Fixed = Chroma.from_documents(texts_Channels_Fixed, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Channels_Fixed_db")
vectordb_Channels_Fixed.persist()

loader_Channels_TimeVarying = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_Channels_TimeVarying.txt')
documents_Channels_TimeVarying = loader_Channels_TimeVarying.load()
texts_Channels_TimeVarying = text_splitter.split_documents(documents_Channels_TimeVarying)
vectordb_Channels_TimeVarying = Chroma.from_documents(texts_Channels_TimeVarying, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Channels_TimeVarying_db")
vectordb_Channels_TimeVarying.persist()


loader_Optimization_SE = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_Optimization_SE.txt')
documents_Optimization_SE = loader_Optimization_SE.load()
texts_Optimization_SE = text_splitter.split_documents(documents_Optimization_SE)
vectordb_Optimization_SE = Chroma.from_documents(texts_Optimization_SE, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Optimization_SE_db")
vectordb_Optimization_SE.persist()

loader_Optimization_EE = TextLoader('/home/student/zzc/代码/GAI-agent-satellite-main/database/ref_Optimization_EE.txt')
documents_Optimization_EE = loader_Optimization_EE.load()
texts_Optimization_EE = text_splitter.split_documents(documents_Optimization_EE)
vectordb_Optimization_EE = Chroma.from_documents(texts_Optimization_EE, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Optimization_EE_db")
vectordb_Optimization_EE.persist()


for chunk_size in chunk_sizes:
    rr_results[chunk_size] = {}
    for chunk_number in chunk_numbers:
        # 1. 重新创建 TextSplitter 和 Retriever
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)

        texts_Scenarios_Homogeneous = text_splitter.split_documents(documents_Scenarios_Homogeneous)
        vectordb_Scenarios_Homogeneous = Chroma.from_documents(texts_Scenarios_Homogeneous, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Scenarios_Homogeneous_db")
        vectordb_Scenarios_Homogeneous.persist()
        retrievers_dict["Scenarios_Homogeneous"] = vectordb_Scenarios_Homogeneous.as_retriever(search_kwargs={"k": chunk_number})

        texts_Scenarios_Heterogeneous = text_splitter.split_documents(documents_Scenarios_Heterogeneous)
        vectordb_Scenarios_Heterogeneous = Chroma.from_documents(texts_Scenarios_Heterogeneous, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Scenarios_Heterogeneous_db")
        vectordb_Scenarios_Heterogeneous.persist()
        retrievers_dict["Scenarios_Heterogeneous"] = vectordb_Scenarios_Heterogeneous.as_retriever(search_kwargs={"k": chunk_number})

        texts_SDMA = text_splitter.split_documents(documents_SDMA)
        vectordb_SDMA = Chroma.from_documents(texts_SDMA, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/SDMA_db")
        vectordb_SDMA.persist()
        retrievers_dict["Access Protocols_SDMA"] = vectordb_SDMA.as_retriever(search_kwargs={"k": chunk_number})

        texts_RSMA = text_splitter.split_documents(documents_RSMA)
        vectordb_RSMA = Chroma.from_documents(texts_RSMA, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/RSMA_db")
        vectordb_RSMA.persist()
        retrievers_dict["Access Protocols_RSMA"] = vectordb_RSMA.as_retriever(search_kwargs={"k": chunk_number})

        texts_Channels_Fixed = text_splitter.split_documents(documents_Channels_Fixed)
        vectordb_Channels_Fixed = Chroma.from_documents(texts_Channels_Fixed, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Channels_Fixed_db")
        vectordb_Channels_Fixed.persist()
        retrievers_dict["Channel Models_Fixed"] = vectordb_Channels_Fixed.as_retriever(search_kwargs={"k": chunk_number})

        texts_Channels_TimeVarying = text_splitter.split_documents(documents_Channels_TimeVarying)
        vectordb_Channels_TimeVarying = Chroma.from_documents(texts_Channels_TimeVarying, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Channels_TimeVarying_db")
        vectordb_Channels_TimeVarying.persist()
        retrievers_dict["Channel Models_Time-Varying"] = vectordb_Channels_TimeVarying.as_retriever(search_kwargs={"k": chunk_number})

        texts_Optimization_SE = text_splitter.split_documents(documents_Optimization_SE)
        vectordb_Optimization_SE = Chroma.from_documents(texts_Optimization_SE, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Optimization_SE_db")
        vectordb_Optimization_SE.persist()
        retrievers_dict["Optimization Goals_SE"] = vectordb_Optimization_SE.as_retriever(search_kwargs={"k": chunk_number})

        texts_Optimization_EE = text_splitter.split_documents(documents_Optimization_EE)
        vectordb_Optimization_EE = Chroma.from_documents(texts_Optimization_EE, embeddings, persist_directory="/home/student/zzc/代码/GAI-agent-satellite-main/database/Optimization_EE_db")
        vectordb_Optimization_EE.persist()
        retrievers_dict["Optimization Goals_EE"] = vectordb_Optimization_EE.as_retriever(search_kwargs={"k": chunk_number})

        
        # 2. 计算检索率
        rr = evaluate_retrieval_rate(annotation_queries, retrievers_dict, block_representation_embeddings, sub_block_representation_embeddings, pipe, embeddings)
        rr_results[chunk_size][chunk_number] = rr
        print(f"Chunk Size: {chunk_size}, Chunk Number (K): {chunk_number}, Retrieval Rate: {rr:.2f}%")

# --- 绘制折线图 ---
plt.figure(figsize=(10, 6))

for k_value in chunk_numbers:
    chunk_rr_values = [rr_results[chunk_size][k_value] for chunk_size in chunk_sizes]
    plt.plot(chunk_sizes, chunk_rr_values, label=f'K={k_value}', marker='o')

plt.xlabel('The size of chunks')
plt.ylabel('Retrieval Rate (%)')
plt.title('Retrieval Rate vs. Chunk Size and Chunk Number (K)')
plt.xticks(chunk_sizes)
plt.ylim(0, 100)
plt.legend(title='Chunk Number (K)')
plt.grid(True)
plt.savefig("/home/student/zzc/代码/GAI-agent-satellite-main/database/result.png")
plt.show()
