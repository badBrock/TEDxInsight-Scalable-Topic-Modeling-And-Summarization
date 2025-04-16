# app.py
from flask import Flask, render_template, request, jsonify, session
import traceback
import pandas as pd
import re
from collections import defaultdict
import os
import json
import pickle
import traceback
import logging
from threading import Lock
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from classic import TEDCommentAnalyzer
import requests
import random

DEBUG_LOG_FILE = "debug_log.txt"
# Add this near your other global variables
analysis_results = {}
# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Flask
app = Flask(__name__)
app.secret_key = 'ted_comment_analyzer_secret_key'

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "merge_comments", "E:\\NLP_Recommender_System\\merge_comments\\video_comments.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Thread lock for model initialization
model_lock = Lock()
analyzer = None

def get_analyzer():
    """Get or initialize the analyzer instance with thread safety"""
    global analyzer
    if analyzer is None:
        with model_lock:
            if analyzer is None:  # Double-check locking
                try:
                    logger.info(f"Creating analyzer with CSV path: {CSV_PATH}")
                    analyzer = TEDCommentAnalyzer(CSV_PATH)
                except Exception as e:
                    logger.error(f"Failed to initialize analyzer: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
    return analyzer

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process the video analysis request"""
    global analysis_results
    video_id = request.form.get('video_id', '').strip()
    
    logger.info(f"Received analysis request for video ID: {video_id}")
    
    if not video_id:
        logger.warning("Empty video ID submitted")
        return jsonify({'error': 'Please enter a valid video ID.'}), 400
    
    try:
        # Get analyzer instance
        analyzer = get_analyzer()
        
        # Analyze video
        topic_model, topic_to_docs, topic_df = analyzer.analyze_video_by_id(video_id)
        
        # Extract topic keywords
        topic_keywords = analyzer.extract_topic_keywords(topic_model)
        
        # Convert topic_df to dict for JSON serialization
        topic_df_dict = topic_df.to_dict(orient='records')
        
        # Convert defaultdict to dict and ensure all values are JSON serializable
        topic_docs_dict = {str(k): v for k, v in topic_to_docs.items()}
        
        # Store only the video_id in session for reference
        session['video_id'] = video_id
        
        # Store full results in global dictionary
        analysis_results[video_id] = {
            'topic_df': topic_df_dict,
            'topic_to_docs': topic_docs_dict,
            'topic_keywords': topic_keywords
        }
        
        # Get available topics (excluding -1 which is the "no topic" category)
        available_topics = sorted([int(t) for t in topic_to_docs.keys() if int(t) != -1])
        
        logger.info(f"Analysis complete. Found {len(available_topics)} topics")
        
        # Return success with initial data
        return jsonify({
            'success': True,
            'topic_df': topic_df_dict,
            'available_topics': available_topics
        })
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Analysis error: {str(e)}"}), 500

@app.route('/get_comments_by_topic/<topic_id>')
def get_comments_by_topic(topic_id):
    """Get comments for a specific topic"""
    global analysis_results
    logger.info(f"Fetching comments for topic ID: {topic_id}")
    
    try:
        topic_id_int = int(topic_id)
        video_id = session.get('video_id')
        
        if not video_id:
            logger.warning("No video ID found in session")
            return jsonify({'error': 'No video has been analyzed. Please analyze a video first.'}), 400
            
        if video_id not in analysis_results:
            logger.warning(f"No analysis results found for video ID: {video_id}")
            return jsonify({'error': 'Analysis results not found. Please analyze the video again.'}), 404
            
        topic_to_docs = analysis_results[video_id]['topic_to_docs']
        
        # Look for the topic as a string since we stored it that way
        if str(topic_id_int) not in topic_to_docs:
            logger.warning(f"Topic {topic_id_int} not found in available topics")
            return jsonify({'error': f'Topic {topic_id_int} not found'}), 404
        
        # Get comments for the requested topic
        comments = topic_to_docs[str(topic_id_int)]
        
        logger.info(f"Returning {len(comments)} comments for topic {topic_id_int}")
        
        return jsonify({
            'topic': topic_id_int,
            'comments': comments,
            'count': len(comments)
        })
    
    except ValueError:
        logger.error(f"Invalid topic ID format: {topic_id}")
        return jsonify({'error': 'Invalid topic ID format'}), 400
    except Exception as e:
        logger.error(f"Error fetching comments by topic: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/generate_topic_name/<topic_id>')
def generate_topic_name(topic_id):
    """Generate a human-readable name for a topic using Mistral"""
    global analysis_results
    logger.info(f"Generating name for topic ID: {topic_id}")
    
    try:
        topic_id_int = int(topic_id)
        video_id = session.get('video_id')
        
        if not video_id:
            return jsonify({'error': 'No video has been analyzed. Please analyze a video first.'}), 400
            
        if video_id not in analysis_results:
            return jsonify({'error': 'Analysis results not found. Please analyze the video again.'}), 404
        
        topic_to_docs = analysis_results[video_id]['topic_to_docs']
        topic_keywords = analysis_results[video_id]['topic_keywords']
        
        if str(topic_id_int) not in topic_to_docs:
            return jsonify({'error': f'Topic {topic_id_int} not found'}), 404
        
        # Get keywords for this topic
        keywords = topic_keywords.get(topic_id_int, ["unknown"])
        
        # Call the name generator function
        analyzer = get_analyzer()
        topic_name = analyzer.name_topic_with_mistral(
            topic_id_int, 
            {int(k): v for k, v in topic_to_docs.items()},  # Convert keys to int
            keywords, 
            num_comments=3
        )
        
        return jsonify({
            'topic': topic_id_int,
            'name': topic_name
        })
        
    except ValueError:
        return jsonify({'error': 'Invalid topic ID format'}), 400
    except Exception as e:
        logger.error(f"Error generating topic name: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/summarize_topic/<topic_id>')
def summarize_topic(topic_id):
    """Generate a summary of comments for a specific topic"""
    global analysis_results
    logger.info(f"Summarizing topic ID: {topic_id}")
    
    try:
        topic_id_int = int(topic_id)
        video_id = session.get('video_id')
        
        if not video_id:
            return jsonify({'error': 'No video has been analyzed. Please analyze a video first.'}), 400
            
        if video_id not in analysis_results:
            return jsonify({'error': 'Analysis results not found. Please analyze the video again.'}), 404
        
        topic_to_docs = analysis_results[video_id]['topic_to_docs']
        
        if str(topic_id_int) not in topic_to_docs:
            return jsonify({'error': f'Topic {topic_id_int} not found'}), 404
        
        # Call the summarizer function
        analyzer = get_analyzer()
        summary = analyzer.summarize_single_topic(
            str(topic_id_int),  # Pass as string since keys are strings
            topic_to_docs,
            num_samples=5
        )
        
        return jsonify({
            'topic': topic_id_int,
            'summary': summary
        })
        
    except ValueError:
        return jsonify({'error': 'Invalid topic ID format'}), 400
    except Exception as e:
        logger.error(f"Error summarizing topic: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/summarize_transcript/<video_id>')
def summarize_transcript(video_id):
    """Generate a summary of the TED Talk transcript"""
    logger.info(f"Summarizing transcript for video ID: {video_id}")
    
    try:
        analyzer = get_analyzer()
        logger.info("Analyzer instance obtained successfully")
        
        # Add more detailed logging
        logger.info("Starting transcript summary generation")
        
        # Check if video exists in data
        if video_id not in analyzer.data['video_id'].values:
            error_msg = f"Video ID {video_id} not found in database"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 404
        
        summary = analyzer.get_transcript_summary(video_id)
        
        if summary.startswith("Error:"):
            logger.error(f"Summary generation failed: {summary}")
            return jsonify({'error': summary}), 500
            
        logger.info(f"Summary generated successfully: {summary[:50]}...")
        
        return jsonify({
            'video_id': video_id,
            'summary': summary
        })
        
    except Exception as e:
        error_msg = f"Error summarizing transcript: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500
@app.route('/check_transcript/<video_id>')
def check_transcript(video_id):
    """Check if transcript is available for a video"""
    try:
        analyzer = get_analyzer()
        result = analyzer.check_transcript_availability(video_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
if __name__ == '__main__':
    # Run without debug mode to prevent auto-reloading
    app.run(debug=False, threaded=True)