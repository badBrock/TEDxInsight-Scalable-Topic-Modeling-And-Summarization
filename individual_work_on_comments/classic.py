# classic.py
import os
import re
import pickle
import ast
import logging
import traceback
import pandas as pd
from collections import defaultdict
import requests
import json
import random
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken  # for token counting
import concurrent.futures  # for parallel processing
nltk.download('punkt')
logger = logging.getLogger(__name__)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

class TEDCommentAnalyzer:
    def __init__(self, video_csv_path):
        logger.info(f"Initializing TEDCommentAnalyzer with CSV: {video_csv_path}")
        self.video_csv_path = video_csv_path

        if not os.path.exists(video_csv_path):
            logger.error(f"CSV file not found: {video_csv_path}")
            raise FileNotFoundError(f"CSV file not found: {video_csv_path}")

        try:
            self.data = pd.read_csv(video_csv_path)
            logger.info(f"Successfully loaded CSV with {len(self.data)} rows")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "]", flags=re.UNICODE)

        try:
            logger.info("Loading SentenceTransformer model")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            raise

        self.cache = {}
        self._load_cache()

    def _get_cache_path(self, video_id):
        return os.path.join(CACHE_DIR, f"analysis_{video_id}.pkl")

    def _load_cache(self):
        logger.info("Loading cached analysis results")
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.startswith("analysis_") and filename.endswith(".pkl"):
                    try:
                        video_id = filename[9:-4]
                        with open(os.path.join(CACHE_DIR, filename), 'rb') as f:
                            self.cache[video_id] = pickle.load(f)
                        logger.info(f"Loaded cached result for video ID: {video_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load cache file {filename}: {str(e)}")

    def _save_to_cache(self, video_id, results):
        try:
            cache_path = self._get_cache_path(video_id)
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved analysis results to cache for video ID: {video_id}")
            self.cache[video_id] = results
        except Exception as e:
            logger.error(f"Failed to save cache for video ID {video_id}: {str(e)}")

    def clean_comment(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#[\w-]+", "", text)
        text = self.emoji_pattern.sub(r'', text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def analyze_video_by_id(self, video_id):
        logger.info(f"Analyzing video with ID: {video_id}")

        if video_id in self.cache:
            logger.info(f"Using cached results for video ID: {video_id}")
            return self.cache[video_id]

        video_data = self.data[self.data['video_id'] == video_id]
        if video_data.empty:
            logger.error(f"No video found with ID: {video_id}")
            raise ValueError(f"No video found with ID: {video_id}")

        logger.info("Extracting and cleaning comments")
        try:
            all_comments_str = video_data.iloc[0]['all_comments']
            if not isinstance(all_comments_str, str) or all_comments_str.strip() == "":
                logger.error("No comments found for this video or comments field is empty")
                raise ValueError("No comments found for this video")
            comments = [self.clean_comment(c.strip()) for c in all_comments_str.split('_') if c.strip()]
            comments = [c for c in comments if c]
            if not comments:
                logger.error("No valid comments found after cleaning")
                raise ValueError("No valid comments found after cleaning")
            logger.info(f"Extracted {len(comments)} comments after cleaning")
        except Exception as e:
            logger.error(f"Error extracting comments: {str(e)}")
            raise

        logger.info("Generating embeddings")
        try:
            embeddings = self.embedding_model.encode(comments, show_progress_bar=True)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

        logger.info("Setting up topic modeling")
        try:
            umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
            keybert_model = KeyBERTInspired()
            representation_model = {"KeyBERT": keybert_model}

            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                top_n_words=3,
                verbose=True
            )

            logger.info("Fitting topic model")
            topics, probs = topic_model.fit_transform(comments, embeddings)
            logger.info(f"Topic modeling complete. Found {len(set(topics))} topics")

            topic_to_docs = defaultdict(list)
            for doc, topic in zip(comments, topics):
                topic_to_docs[topic].append(doc)

            topic_df = topic_model.get_topic_info()
            logger.info("Analysis completed successfully")

            result = (topic_model, dict(topic_to_docs), topic_df)
            self._save_to_cache(video_id, result)
            return result
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    def name_topic_with_mistral(self, topic_id, topic_comments_dict, topic_keywords_list, num_comments=3):
        """
        Generate a short topic name using Mistral from keywords (in a list) and a few example comments.
        """
        # Safety checks
        if not isinstance(topic_keywords_list, list) or topic_id not in topic_comments_dict:
            return f"Topic {topic_id}"

        try:
            keywords = topic_keywords_list[topic_id]
        except (IndexError, TypeError):
            keywords = ["unknown"]
            
        comments = topic_comments_dict[topic_id]
        sampled_comments = random.sample(comments, min(len(comments), num_comments))

        # Prompt setup
        prompt = (
            "Given the following information, generate a short and clear topic name (2-5 words) that best represents the theme.Choose the best one.\n\n"
            f"Top keywords: {', '.join(keywords)}\n"
            "Sample comments:\n"
            + "\n".join(f"- {c}" for c in sampled_comments) +
            "\n\nTopic name:"
        )

        # API call setup
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }

        # Streaming Mistral response
        topic_name = ""
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(data)
                        delta = json_data["choices"][0]["delta"]
                        if "content" in delta:
                            topic_name += delta["content"]
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Error generating name for topic {topic_id}: {str(e)}")
            topic_name = f"Topic {topic_id}"

        return topic_name.strip()

    def summarize_single_topic(self, topic_id, topic_comments_dict, num_samples=5):
        """
        Given a topic ID and dictionary of {topic_id: [comments]},
        returns a summary of sampled comments from that topic.
        """
        if topic_id not in topic_comments_dict:
            return f"❌ Topic {topic_id} not found."

        comments_list = topic_comments_dict[topic_id]
        sampled = random.sample(comments_list, min(len(comments_list), num_samples))

        # Prompt preparation
        prompt_text = f"Summarize the following user comments into a short paragraph capturing the main sentiments and themes:\n\n"
        prompt_text += "\n".join(f"- {comment}" for comment in sampled)

        # API setup
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": True
        }

        # Streaming response
        summary = ""
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(data)
                        delta = json_data["choices"][0]["delta"]
                        if "content" in delta:
                            summary += delta["content"]
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Error summarizing topic {topic_id}: {str(e)}")
            summary = f"Error summarizing topic {topic_id}: {str(e)}"

        return summary.strip()


    def extract_topic_keywords(self, topic_model):
        """
        Extract keywords for each topic from the topic model.
        Returns a dict: {topic_id: [keywords]}
        """
        topic_info = topic_model.get_topic_info()
        topic_keywords = {}

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                representation = row['Representation']
                if isinstance(representation, str):
                    try:
                        # Convert string to list
                        keywords = ast.literal_eval(representation)
                        topic_keywords[topic_id] = keywords
                    except Exception as e:
                        print(f"Error parsing topic {topic_id}: {e}")
        
        return topic_keywords
    
    def clean_transcript_string(self, transcript):
    # Remove transcription noise like (Applause), [detained], [many]
        transcript = re.sub(r'\[.*?\]|\(.*?\)', '', transcript)

        # Normalize punctuation: keep periods, commas, question marks, exclamations
        transcript = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', transcript)

        # Normalize multiple spaces to single space
        transcript = re.sub(r'\s+', ' ', transcript)

        # Convert to lowercase and strip
        transcript = transcript.lower().strip()

        # Split into sentences based on ., ?, or !
        sentences = re.split(r'[.?!]', transcript)

        # Clean up empty strings and strip extra whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def get_transcript_summary(self, video_id):
        try:
            logger.info(f"Getting transcript for video ID: {video_id}")
            # Find the video in the data
            video_data = self.data[self.data['video_id'] == video_id]
            
            if video_data.empty:
                logger.error(f"No data found for video ID: {video_id}")
                return "No transcript found for this video. Please ensure the video ID is correct."
            
            # Debug: What columns are available in the data?
            logger.info(f"Available columns: {video_data.columns.tolist()}")
            
            # Get the transcript - check if the column exists first
            if 'all_comments' not in video_data.columns:
                logger.error("No 'transcript' column found in data")
                return "Transcript data is not available in the database schema."
            
            transcript = video_data.iloc[0].get("all_comments", None)
            if transcript is None or transcript == "" or pd.isna(transcript):
                logger.error("Transcript is empty or None")
                return "Transcript is not available for this video."
            
            logger.info(f"Transcript type: {type(transcript)}")
            logger.info(f"Transcript preview: {str(transcript)[:100]}...")
            
            # Clean and prepare the transcript text
            if isinstance(transcript, str):
                try:
                    # Try to parse if it's stored as a string representation of a list
                    transcript_data = ast.literal_eval(transcript)
                    if isinstance(transcript_data, list):
                        cleaned_text = " ".join([item['text'] for item in transcript_data if isinstance(item, dict) and 'text' in item])
                    else:
                        cleaned_text = str(transcript_data)
                except (ValueError, SyntaxError) as e:
                    logger.info(f"Using transcript as plain text: {e}")
                    # If not a valid list representation, use as is
                    cleaned_text = transcript
            elif isinstance(transcript, list):
                cleaned_text = " ".join([item['text'] for item in transcript if isinstance(item, dict) and 'text' in item])
            else:
                logger.error(f"Invalid transcript format: {type(transcript)}")
                return f"Invalid transcript format: {type(transcript)}."
            
            # Debug the cleaned text
            logger.info(f"Cleaned text length: {len(cleaned_text)}")
            logger.info(f"Cleaned text preview: {cleaned_text[:100]}...")
            
            # Process the cleaned text
            input_data = self.clean_transcript_string(cleaned_text)
            large_input = " ".join(input_data)
            
            logger.info("Calling summarize_large_text...")
            # Return the summary
            return self.summarize_large_text(large_input)

        except Exception as e:
                logger.error(f"Error summarizing transcript: {str(e)}")
                logger.error(traceback.format_exc())
                return f"Error: {str(e)}"

    def summarize_large_text(self, text):
        """
        Summarize large text input by chunking and using OpenRouter API
        """
        import tiktoken
        from nltk.tokenize import sent_tokenize
        import nltk
        
        # Ensure NLTK punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
            # Constants for API
            API_KEYS = [
                "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",
                "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"
            ]
            MODEL_NAME = "mistralai/mistral-7b-instruct:free"
            CHUNK_TOKEN_LIMIT = 3500
            MAX_OUTPUT_TOKENS = 500
            
            # Prompt template
            PROMPT_TEMPLATE = """
            You are an expert TED Talk summarizer. Read the transcript below and summarize it with the following structure:

            1. **Topic**: What is the main subject or theme of the talk? What is being discussed?
            2. **Purpose**: What is the speaker trying to accomplish with this talk? What is the speaker's goal in sharing this message?
            3. **Key Takeaway**: What is the most important point or lesson from the talk that the audience should remember?

            Transcript:
            """
            
            # Helper function to chunk text semantically
            def semantic_chunk_text(text, tokenizer, max_tokens=CHUNK_TOKEN_LIMIT):
                sentences = sent_tokenize(text)
                chunks = []
                current_chunk = ""
                current_len = 0

                for sentence in sentences:
                    sentence_len = len(tokenizer.encode(sentence))

                    if sentence_len > max_tokens:
                        words = sentence.split()
                        chunk = ""
                        for word in words:
                            if len(tokenizer.encode(chunk + " " + word)) <= max_tokens:
                                chunk += " " + word
                            else:
                                chunks.append(chunk.strip())
                                chunk = word
                        if chunk:
                            chunks.append(chunk.strip())
                    elif current_len + sentence_len <= max_tokens:
                        current_chunk += " " + sentence
                        current_len += sentence_len
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_len = sentence_len

                if current_chunk:
                    chunks.append(current_chunk.strip())

                logger.info(f"Total chunks created: {len(chunks)}")
                return chunks
            
            # Call API function
            def call_openrouter_api(chunk, prompt_template, api_key):
                url = "https://openrouter.ai/api/v1/chat/completions"
                prompt = prompt_template + f"""{chunk}"""

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": MAX_OUTPUT_TOKENS
                }

                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Error with API request: {response.status_code}")
                    return None
            
            # Select a working API key
            def get_api_key():
                for api_key in API_KEYS:
                    test_summary = call_openrouter_api("Test chunk", PROMPT_TEMPLATE, api_key)
                    if test_summary:
                        logger.info(f"Using API key: {api_key[:10]}...")
                        return api_key
                raise Exception("All API keys failed!")
            
            try:
                # Initialize tokenizer
                tokenizer = tiktoken.get_encoding("gpt2")
                
                # Get API key
                api_key = get_api_key()
                
                # Split text into chunks
                text_chunks = semantic_chunk_text(text, tokenizer)
                
                # Process each chunk
                intermediate_summaries = []
                for chunk in text_chunks:
                    result = call_openrouter_api(chunk, PROMPT_TEMPLATE, api_key)
                    if result:
                        intermediate_summaries.append(result)
                
                # Combine intermediate summaries
                final_input = "\n".join(intermediate_summaries)
                
                # Ensure final input isn't too large
                if len(tokenizer.encode(final_input)) > CHUNK_TOKEN_LIMIT:
                    logger.info("Final summary input too large. Chunking final input...")
                    final_input_chunks = semantic_chunk_text(final_input, tokenizer)
                    final_summary_parts = []
                    for final_chunk in final_input_chunks:
                        final_summary_parts.append(call_openrouter_api(final_chunk, PROMPT_TEMPLATE, api_key))
                    final_summary = "\n".join(final_summary_parts)
                else:
                    final_summary = call_openrouter_api(final_input, PROMPT_TEMPLATE, api_key)
                
                return final_summary
            
            except Exception as e:
                logger.error(f"Error in summarize_large_text: {str(e)}")
                logger.error(traceback.format_exc())
                return f"Error generating summary: {str(e)}"
            
    def check_transcript_availability(self, video_id=None):
                """Check if transcripts are available in the data"""
                if 'all_commentss' not in self.data.columns:
                    return {"status": "error", "message": "No transcript column in data"}
                
                transcript_availability = {
                    "total_videos": len(self.data),
                    "videos_with_transcripts": self.data['all_comments'].notna().sum()
                }
                
                if video_id:
                    video_data = self.data[self.data['video_id'] == video_id]
                    if video_data.empty:
                        return {"status": "error", "message": f"Video ID {video_id} not found"}
                    
                    has_transcript = video_data.iloc[0].get('all_comments') is not None and not pd.isna(video_data.iloc[0].get('transcript'))
                    transcript_availability["video_id"] = video_id
                    transcript_availability["has_transcript"] = has_transcript
                
                return transcript_availability
    
        
        # except Exception as e:
        #     logger.error(f"Error in summarize_large_text: {str(e)}")
        #     logger.error(traceback.format_exc())
        #     return f"Error generating summary: {str(e)}"