## ISyE 4803 Project

Weichao Chen wchen408@gatech.edu

Linxi Xiao xiaolinxi@gatech.edu

### Introduction

Recommender system is a system that aims to provide recommendations most relevant to the user. It generally recommends in 3 ways, through collaborative filtering, or content-based filtering, or both. Collaborative filtering clusters users based their past behaviors, and predicts items a user might interest in based on his similarity to other users. Content-based filtering recommends based on a description of the item and a profile of user's preference. [[1](Aggarwal, Charu C. (2016). *Recommender Systems: The Textbook*. Springer. [ISBN](https://en.wikipedia.org/wiki/International_Standard_Book_Number) [9783319296579](https://en.wikipedia.org/wiki/Special:BookSources/9783319296579))]

In this project, we attempt to build a recommender system that utilizes users' digital footprints, such as geographical information, browsing history, to recommend products most relevant to their preferences.

### Project Goal

##### <u>Part 1</u>

We will first build the most basic content-based recommender that predicts user's preference. We will employ the cascade model (Craswell et al., 2008), in which the user examines a list of recommended items from top to the bottom and only selects the first attractive item. The goal of the recommender is to approximate the user preference, while minimizing the regret.

##### <u>Part 2</u>

Whereas the basic cascade model assumes the user only select the first attractive item, this part investigates the case when the user selects all recommended items of their interest. (https://arxiv.org/pdf/1602.03146.pdf)

##### <u>Part 3</u>

Using data generated from sections above, we incorpoate collaborative filtering technique to cluster users based on their past click history and geographical information (population density, poverty level, average income), in an attempt to avoid the cold-start issue, where the recommender have no previous information about the user.

### Dataset

<u>**Product Data**</u>

We used the housing data from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) as our products. This dataset contains features of houses in granular detail. Besides basic information such as area and building type, it has attributes that are decomposed from pictures and housing description, such as roof style, heating, garage finish, mimicking the scenario when a user reads through the listing. 

<u>**User Profile**</u>

We generate the user profile using the census data provided by Michael and Dr. Gupta. Each data point in the census dataset is used as an user, and the medium income of the data point is treated as the use's income. User's preferences on housing was generated randomly using housing attributes in product dataset. 

### Part 1: Basic Cascade Model

<u>**Model**</u>

* The recommender have $N$ available products and displays a list of $ S \subsetneq N$ items each time.
* Each product $p_{i}$ has an attribute vector $A_{i}$, all of which are binary attributes. Continuous attributes will be converted into binary using piecewise transformation. For example, \$0-5000, \$5000-10000, etc.. Each attribute vector $A_{i}$ has an associated weight vector $w_{i}$.
* Each user comes in with a set of critieria $C$. Each user proxy will be drew from the census data, with user's critieria set $C$ generated according to user's income level. 
* The recommender aims to recommend items for a particular user and find that user's criteria set $C$.
* On infering user location: although the state-of-the-art *HTML5 Geolocation API* can capture the latitude-longtitude information with user's **explicit** approval, we assume that users are in general unwilling to disclose such information. An alternative, *Reverse IP Lookup* can assit RAs to pinpoint users to a *County* level in most cases.

<u>**Algorithm**</u>

- initialization: $w_{i}^{1} $ = 1 . $ \forall i = 1, ..., N$;

- for $ t $ = 1 to $ T $ do

  - approximate user profile $C_{t} = \sum_{i=1}^{N} w_{i}^{t} A_{i} $ 
  - let $b_{t}$ be a Bernoulli random variable that equals to 1 with probability δ
  - if $b_{t} $ = 1 then
    - choose $S$ items from $N$ items ~ Binomial($\frac{1}{N}$)
    - sort $S$ items in descending order based their cosine similarity with $C_{t}$
    - if item $S_{m}$ is clicked,  $l_{t}(m)$ = $-\frac{1}{m}$ 
    - $l_{t}(i)$ = $\frac{1}{i}$ $\forall i = 0, ..., m-1$
    - $l_{t}(i)$ = $\frac{1}{i}$ $\forall i = m+1, ..., |S|$
    - update $w_{i}^{t+1} = w_{i}^{t}(1-\epsilon)^{l_{t}(i)}$

  - else
    - choose $S$ items based on probability distribution of their weights
    - no update on distribution
  - end if

- end for

<u>**Implementation**</u>

https://github.com/wchen408/4803RA



### Goal

Investigate whether exploiting users' digital finger printings  (geographic, browser, OS, device) helps Recommendation Agent (RA) make better recommendations

### Assumption

* The website sells $N$ products and displays a subset $S \subsetneq N$ each time. $|S| = 10$ 
* Each user come in with the intention to purchase a product that satisfy a set of critieria $C$. Each user proxy will be drew from the census data, with his critieria set $C$ generated from according to his income level. 
* Website sells product(s) that satisfy $C$

### Procedure

Two recommendation engines $RA_1$, $RA_2$. Where $RA_2$ takes in extra degree of information in generating subset $S$ and $RA_1$ does not. 

$t = 0$: 

​	User apply intial filter

For $t = 1...j$ where $\nexists \mbox{ Product P} \in S_{t} \mbox{ s.t. } P \mbox{ satisfies } C $: 

​	User clicks on an item $P_{t}\prime$ most closetly resembles $P$, the item he is looking for. RA readjust weights of all product based on this click pattern and regenerate subset $S_{t+1}$

At $t = j​$: 

​	$\exists \mbox{ Product P} \in S_{t} \mbox{ s.t. } P \mbox{ satisfies } C$

Compare converge time of $RA_1$ $RA_2$

