# **Project Title:** Sentiment Analysis on Rival Universities

# **Project Introduction:**
## **Topic:** Sentiment analysis & comparison between VT and UVA
The pursuit of higher education is a common step in many peoples lives. Our goal is to evaluate and compare the students' and community members' feelings towards Virginia Tech and the University of Virginia; both of which are arguably the most prestigious public universities in Virginia. This includes analyzing online discussions and aggregating sentiments related to academics, campus life, sports, dining, and overall university experiences.

The underlying question is: **How do students and others perceive and express their opinions about these universities on digital platforms?**

One other thing to mention about this project is that all the natural language processing (NLP) was done by hand. This includes manually building the TFIDF matrix and cosine similarity. Some of the machine learning done later in the analysis including clustering and dimension reduction were done using packages. The reason that I did this was the get a better understanding of how NLP works. In the future, I would love to redevlop this project with more advanced techniques using deep learning and AI

## **Sources**
For our research, I'll focus on public online forums that can be scraped and analyzed as datasets. The types of information we are looking for are individual comments and statements regarding UVA or VT from students, alumni, and the general public.

A few sources I used were:
- Reddit: https://www.reddit.com/
- RateMyProfessors: https://www.ratemyprofessors.com/


# **Question**
There has always been this debate, which school is better? The University of Virginia or Virginia Polytechnic Institute and State University? Maybe we should actually find out which one is best. The big question: **Is Virginia Tech better than UVA?**

But, how do we define "better"? We know the rankings of each school based on different websites such as USNews, WSJ, Niche, etc. We could aggregate the average ranking of each source and see who has the higher ranking. The biggest problem with this approach is we don't consider students. These websites are made by reviewers who aren't students or alumni of these universities and many of them don't include a student's experience. This leads us to a better question. **How do students feel about different parts of their university experience at Virginia Tech and UVA?**

# **Analysis Q1**
Throughout this initial analysis, we are going to use comments from Reddit and RateMyProfessors. In choosing these sources,  we are hoping to represent a range of views and opinions from aspiring, current, and former students for each respective university. All of our data will be from the past 10 years, as this gives a relevant representation. The data scraped from these websites will be stored in .csv files for code reproducability purposes. This way we do not have to rescrape every time we rerun our code.

<img src="/Users/mattanderson/Documents/Spring25/3654/Project/Github Upload/images/posnegbar.png" width="550" height="400">

Looking at the sentiment analysis of VT and UVA on Reddit, we see that UVA and VT have very similar sentiments of both being roughly 2.5 times more positive than negative. On the other hand, we see a major difference in the RateMyProfessors data.

# **Conclusion Q1**

The question still stands, **Is Virginia Tech better than UVA?**

From the results of our sentiment analysis and looking at the visuals, our answer is leaning in favor of Virginia Tech being better than UVA. Like described above, the sentiment rates from the Reddit posts are about equal, but Virginia Tech's RateMyProfessors comments are significantly more positive than UVA's.

That being said, there are some limitations we have found in our current analysis. For starters, we see in the word clouds that the Reddit data can still be cleaned further. For example, we plan to include additional stopwords in the file since there are still some filler words that could be filtered out to ensure we have a more accurate representation of both schools. One of the words we are seeing a lot of is "i'm", which in general has a neutral connotation.

In the next iteration of this analysis, we are planning on collecting data from more sources, process the data from Niche, and use the term frequency-inverse document frequency (TFIDF) method to further analyze the relationships between the terms.

# **Question 2**
Virginia Tech and UVA are undoubtedly some of the best schools in the State of Virginia. In our previous analysis we concluded by the metric of student satisfaction, that VT seems to be the better school. In this analysis, we examine Virginia Tech and the University of Virginia under the assumption that each institution excels in different aspects of student life. Our central research question is: **"How do VT and UVA compare across various dimensions?"**

 For this analysis we will be looking at 10 different areas of student life:
 Food, Facilities, Reputation, Happiness, Safety, Opportunities, Clubs, Social, Internet and Location

 - But, how do we compare the schools? We know the rankings of each school based on different websites such as USNews, WSJ, Niche, etc. We could aggregate the average ranking of each source and see who has the higher ranking. The biggest problem with this approach is we don't consider students. These websites are made by reviewers who aren't students or alumni of these universities and many of them don't include a student's experience. This leads us to a better question.

 - After looking at the difference in sentiment in both the Reddit and RateMyProfessors data, we didn't see much of a difference within the Reddit comments. After looking closer at this data, there weren't many reviews, but mostly just students asking questions. This led us to only use the RateMyProfessors data along with categorical data that is also provided on this website.

# **Analysis**
In an effort to gain more insight into the RateMyProfessor comments, we went ahead and extracted the overall rating attached to the comment. With this we are hoping to explore if there is a relationship between the sentiment of the comment and rating itself.

Starting off, we used Selenium to scrap these reviews and attach them to the same comments we used in our inital analysis. This data is then stored in "VTRMPNEW.csv" and "UVARMPNEW.csv".


## Visualizations
With all of the data imported and cleaned, we can now begin visualization. First, we'll use a parallel coordinates graph to identify patterns or correlations between the 8 different aspects of student life across both schools.

<img src="/Users/mattanderson/Documents/Spring25/3654/Project/Github Upload/images/parallelcords.png" width=1000 height=300>

The parallel coordinates graph above displays ratings for nine aspects of students life on a scale from 0-5. The lines indicate rating combinations, with the maroon representing VT and navy blue representing UVA.
Some things we observe are:
- For most categories, ratings range from 1-5 for both schools. This suggests that students are having varied experiences at their institution.
- When looking at the saftey ratings, we see that UVA has several low ratings that drop to 0. This suggests that UVA students may have greater saftey concerns than VT students.
- While there is a considerable overlap between schools, VT seems to have more ratings in the 3-5 range for Reputation and Food, and slightly higher ratings for Opportunities.

Next, we'll examine the Principal Component Analysis (PCA) and Multidimensional Scaling (MDS) of these attributes.

<img src="/Users/mattanderson/Documents/Spring25/3654/Project/Github Upload/images/pca.png" width="550" height="400">

Looking at the PCA graph, we observe overlap between schools in the center-right region, with VT's points clustering more tightly in the upper right section. This suggests VT students may have more consistent experiences. UVA's points are more dispersed, extending toward the left and lower regions, which could indicate greater variability in their student experiences.

<img src="/Users/mattanderson/Documents/Spring25/3654/Project/Github Upload/images/mds.png" width="550" height="400">

Looking at the MDS graph, we observe a similar pattern to the PCA, but with clearer separation. VT's points predominantly occupy the upper regions of the graph, while UVA's points are more concentrated in the lower region. We also notice a distinct cluster of UVA points at the bottom of the graph. This vertical differentiation suggests a significant difference in how students at each institution perceive and rate their school.

## Classification
Lastly, we were curious if we could build a classifier that would correctly predict if comments are directed towards VT or UVA.

First, we want to build a TFIDF matrix on the bag including all comments from each school. This allows us to look at the key words associated with each school.

Here we can see that both schools have key words that reflect varying views and ideals.

For UVA:
- "competitive" and "exclusive" - suggests an elite and selective academic or social environment
- "special" and "favorite" - suggests strong, positive connections to aspects of the university
- "doubt" - suggests students potentially struggling with the competitive and exclusive environment

For VT:
- "safe," "balanced," and "prepared" - suggests either a focus on practical education and future success or the well-being of the community
- "proactive" and "ambitious" - suggests a forward thinking, driven student body
- "abuse" and "expensive" - suggests concerns about cost and negative campus experiences

Now we will develop our classification model. We'll create a TF-IDF matrix from student comments, combine text features with numerical rating data, and train a classifier. We've chosen to implement a Decision Tree classifier due to its versatility in handling both numerical and categorical data.

Our classifier accurately predicted the school associated with each comment. It identified UVA comments with 92% precision and 92% recall, and VT comments with 92% precision and 92% recall.

# **Conclusion**

Let's start by addressing our focused question: **"How do Virginia Tech and UVA compare across various dimensions?"**

Based on our comprehensive analysis, each technique revealed different insights. Our cosine similarity analysis showed that some comment pairs had perfect similarity scores despite representing different schools, indicating students from both institutions share similar sentiments. The parallel coordinates graph demonstrated that while both schools received varied ratings across the 10 categories, UVA students expressed greater safety concerns, while VT had higher ratings for reputation, food quality, and opportunities. The PCA and MDS revealed significant differences in student experiences, with VT showing more consistent ratings and UVA displaying greater variability, suggesting VT may offer a more uniform student experience.

Now let's circle back to our original question: **"Is Virginia Tech better than UVA?"**

While this will always be a subjective question, our analyses from both parts of the assignment suggest Virginia Tech receives more positive sentiments and provides a more well-rounded, stable environment. This is evidenced in both the PCA and MDS outcomes as well as the keywords associated with each institution.
