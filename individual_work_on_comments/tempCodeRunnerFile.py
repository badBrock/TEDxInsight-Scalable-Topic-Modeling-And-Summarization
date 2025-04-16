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
    