## TEXT-SUMMARIZATION-TOOL


*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: HASHMI SYED FAHAD

*INTERN ID*: CT08DL815

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 8 WEEEKS

*MENTOR*: NEELA SANTOSH



## Description :

## Project Overview :-

As a student passionate about natural language processing, I created bart-text-summarizer, a Python script that harnesses the BART model (facebook/bart-large-cnn) from Hugging Face to transform lengthy articles into concise, coherent summaries. My earlier experiments with LSTM models were a struggle—producing garbled text like “musensiblet” and hitting my laptop’s memory ceiling at ~1025 MB. Determined to build something reliable and portfolio-worthy for AI internship applications, I developed this tool to summarize complex texts, like an article on AI’s impact, and save the results for sharing. This project reflects my growth in NLP, coding, and problem-solving, and I’m excited to share what I’ve learned.


## What It Does :-

This script uses BART’s pre-trained capabilities to generate summaries of 50-150 words from texts up to 1024 tokens. It features a sample article about AI’s role in healthcare, finance, and education, producing a distilled summary that captures key points. The output is formatted cleanly with textwrap, printed to the console, and saved to summary_output.txt. Robust error handling ensures the script doesn’t crash on invalid inputs, and a lighter model option (facebook/bart-base) accommodates memory constraints. Designed for versatility, it’s easy to customize with your own articles, making it a practical tool for research, content creation, or learning.


## Lesson Learned :-

A transformative experience emerged with this project after solving LSTM failures which had caused incorrect results through extensive training. The  pre-trained BART model brought about a groundbreaking moment by producing precise summaries through its default configuration. I acquired  practical experience with Hugging Face’s Transformers library by working with BartTokenizer to handle text input and BartForConditionalGeneration to produce summaries. The practice of adjusting parameters such as num_beams=4 and length_penalty=2.0 enabled me to achieve the right balance between concise brevity and clear communication which will benefit my  NLP work in the future.  The management of computer memory posed a recurring problem. My laptop with  1025 MB RAM capacity could not handle BART's 1-2 GB requirements so I deployed facebook/bart-base which uses 800 MB and experimented with shorter texts to maintain system performance. I improved my hardware optimization abilities through  practical experience which I now apply to real-world AI projects. The development of robust code marked a significant achievement as  I implemented input validation together with error handling which protected users from script crashes when dealing with missing texts.  The  experience of writing this README along with detailed code explanations helped me develop the ability to present complex technical concepts  in a way that would be understood by my peers. After summarizing an article about AI's practical applications  I discovered the strength of NLP which led me to develop research summary ideas.



## Requirements:-

1. torch>=1.9.0
2. transformers>=4.20.0



## Setup and Usage :-

Install Dependencies >

```

pip install torch transformers
```

Or use >

```
pip install -r requirements.txt
```



Run >

```

python text_summarizer.py

```

Customize: Modify article in text_summarizer.py or import summarize_text >

```

from text_summarizer import summarize_text
my_text = "Your article here..."
print(summarize_text(my_text))

```


## Notes:-

Memory: Use model_name="facebook/bart-base" for ~800 MB if ~1-2 GB is too much.



Performance: ~5-10 seconds on CPU.



Next Steps: Add a CLI or multilingual support.



## OUTPUT:-

![Image](https://github.com/user-attachments/assets/c19f727d-b942-400d-9136-79a6c00a3171)
