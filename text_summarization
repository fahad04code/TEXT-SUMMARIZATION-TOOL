import textwrap
from transformers import BartForConditionalGeneration, BartTokenizer
import sys

def summarize_text(input_text, max_length=150, min_length=50, model_name="facebook/bart-large-cnn"):
    """
    Summarizes input text using the BART model.
    
    Args:
        input_text (str): Text to summarize.
        max_length (int): Maximum summary length in tokens.
        min_length (int): Minimum summary length in tokens.
        model_name (str): Hugging Face BART model name.
    
    Returns:
        str: Summary or error message.
    """
    # Validate input
    if not isinstance(input_text, str):
        return f"Error: Input must be a string, got {type(input_text)}"
    if not input_text.strip():
        return "Error: Input text is empty"
    
    try:
        # Load tokenizer and model
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Encode input text
        inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    except Exception as e:
        return f"Error: Failed to summarize text; {str(e)}"

def main():
    # Sample article (AI in industries)
    article = """
    Artificial intelligence (AI) is reshaping industries across the globe, driving innovation in sectors like healthcare, finance, and education. In healthcare, AI-powered tools analyze medical images, such as X-rays and MRIs, with precision that matches or exceeds human experts, enabling early detection of diseases like cancer. For example, recent studies show AI algorithms can identify breast cancer in mammograms with 95% accuracy, improving patient outcomes through timely interventions. In finance, AI enhances efficiency through algorithmic trading, fraud detection, and personalized banking. Banks use machine learning to monitor transactions in real-time, flagging suspicious activities with minimal false positives. In education, AI-driven platforms personalize learning, adapting content to students’ needs. However, AI’s rise sparks challenges, including ethical concerns like algorithmic bias, job displacement, and data privacy. For instance, biased AI models in hiring can unfairly screen candidates. Policymakers are working to create regulations that balance innovation with fairness, while researchers aim to develop transparent AI systems. As AI integrates deeper into daily life, its long-term impact remains a topic of intense debate, with experts calling for responsible development to ensure benefits outweigh risks.
    """
    
    # Display original text
    print("Original Article:")
    print(textwrap.fill(article, width=80))
    print("\nSummary:")
    
    # Generate and display summary
    summary = summarize_text(article)
    if "Error" not in summary:
        print(textwrap.fill(summary, width=80))
    else:
        print(summary)
    
    # Save output to file
    try:
        with open("summary_output.txt", "w", encoding="utf-8") as f:
            f.write("Original Article:\n")
            f.write(textwrap.fill(article, width=80))
            f.write("\n\nSummary:\n")
            f.write(textwrap.fill(summary, width=80) if "Error" not in summary else summary)
        print("\nSaved output to 'summary_output.txt'")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running script: {e}")
        sys.exit(1)
