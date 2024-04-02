# DataLLM: _prompt LLMs for Tabular Data_ 🔮

Welcome to DataLLM, your go-to open-source platform for Tabular Data Generation!

DataLLM allows you to efficiently tap into the vast power of LLMs to...
1. **create mock data** that fits your needs, as well as
2. **enrich datasets** with world knowledge.

MOSTLY AI is hosting a rate-limited DataLLM server instance at [data.mostly.ai](https://data.mostly.ai), with its default model being a fine-tuned Mistral-7b.

_Note_: For use cases, that relate to personal data, it is advised to enrich synthetic rather than real data. 
For one, to protect the privacy of the individuals when transmitting data. 
And for two, to preempt ethical concerns regarding the application of LLMs on personal data. 
This principle is referred to as **Synthetic Data Augmented Generation (SAG)**, which taps into the knowledge of LLMs but grounded in a representative version of your data assets. MOSTLY AI is hosting a top-in-class Synthetic Data platform at https://app.mostly.ai, which allows you to easily transform your existing customer data into synthetic data.


## Start using DataLLM

1. Sign in and retrieve your API key [here](https://data.mostly.ai/docs/routes#authentication).
2. Install the latest version of the DataLLM Python client.
```shell
pip install -U datallm
```
3. Instantiate a client with your retrieved API key.
```python
from datallm import DataLLM
datallm = DataLLM(api_key='INSERT_API_KEY', base_url='https://data.mostly.ai')
```
4. Enrich an existing dataset with new columns, that are coherent with any of the already present columns.
```python
import pandas as pd
df = pd.DataFrame({
    "age in years": [5, 10, 13, 19, 30, 40, 50, 60, 70, 80],
    "gender": ["m", "f", "m", "f", "m", "f", "m", "f", "m", "f"],
    "country code": ["AT", "DE", "FR", "IT", "ES", "PT", "GR", "UK", "SE", "FI"],
})

# enrich the DataFrame with a new column containing the official country name
df["country"] = datallm.enrich(df, prompt="official name of the country")

# enrich the DataFrame with first name and last name
df["first name"] = datallm.enrich(df, prompt="the first name of that person")
df["last name"] = datallm.enrich(df, prompt="the last name of that person")

# enrich the DataFrame with a categorical
df["age group"] = datallm.enrich(
    df, prompt="age group", categories=["kid", "teen", "adult", "elderly"]
)

# enrich with a boolean value and a integer value
df["speaks german"] = datallm.enrich(df, prompt="speaks german?", dtype="boolean")
df["body height"] = datallm.enrich(df, prompt="the body height in cm", dtype="integer")
print(df)
#    age in years gender country code         country first name   last name age group speaks german  body height
# 0             5      m           AT         Austria     Julian     Kittner       kid          True          106
# 1            10      f           DE         Germany      Julia     Buchner      teen          True          156
# 2            13      m           FR          France   Benjamin    Dumoulin      teen         False          174
# 3            19      f           IT           Italy    Alessia  Santamaria      teen         False          163
# 4            30      m           ES           Spain       Paco        Ruiz     adult         False          185
# 5            40      f           PT        Portugal      Elisa      Santos     adult         False          168
# 6            50      m           GR          Greece   Dimitris     Kleopas     adult         False          166
# 7            60      f           UK  United Kingdom      Diane     Huntley   elderly         False          162
# 8            70      m           SE          Sweden       Stig   Nordstrom   elderly         False          174
# 9            80      f           FI         Finland       Aili      Juhola   elderly         False          157
```
5. Or create a completely new dataset from scratch.
```python
df = datallm.mock(
    n=100,  # number of generated records 
    data_description="Guests of an Alpine ski hotel in Austria",
    columns={
        "full name": {"prompt": "first name and last name of the guest"},
        "nationality": {"prompt": "the 2-letter code for the nationality"},
        "date_of_birth": {"prompt": "the date of birth of that guest", "dtype": "date"},
        "gender": {"categories": ["male", "female", "non-binary", "n/a"]},
        "beds": {"prompt": "the number of beds within the hotel room; min: 2", "dtype": "integer"},
        "email": {"prompt": "the customers email address", "regex": "([a-z|0-9|\\.]+)(@foo\\.bar)"},
    },
    temperature=0.7
)
print(df)
#            full name nationality date_of_birth  gender  beds                   email
# 0     Melinda Baxter          US    1986-07-09  female     2   melindabaxter@foo.bar
# 1         Andy Rouse          GB    1941-03-14    male     4       andyrouse@foo.bar
# 2      Andreas Kainz          AT    2001-01-10    male     2   andreas.kainz@foo.bar
# 3         Lisa Nowak          AT    1994-01-02  female     2       lisanowak@foo.bar
# ..               ...         ...           ...     ...   ...                     ...
# 96     Mike Peterson          US    1997-04-28    male     2    mikepeterson@foo.bar
# 97    Susanne Hintze          DE    1987-04-12  female     2         shintze@foo.bar
# 98  Ernst Wisniewski          AT    1992-04-03    male     2  erntwisniewski@foo.bar
# 99    Tobias Schmitt          AT    1987-06-24    male     2  tobias.schmitt@foo.bar
```

## Key Features

- **Efficient Tabular Data Generation:** Easily prompt LLMs for structured data at scale.
- **Contextual Generation:** Each data row is sampled independently, and considers the prompt, the existing row values, as well as the dataset descriptions as context.
- **Data Type Adherence:** Supported data types are `string`, `categorical`, `integer`, `floats`, `boolean`, `date`, and `datetime`.
- **Regular Expression Support:** Further constrain the range of allowed values with regular expressions.
- **Flexible Sampling Parameters:** Tailor the diversity and realism of your generated data via `temperature` and `top_p`.
- **Esay-to-use Python Client:** Use `datallm.mock()` and `datallm.enrich()` directly from any Python environment.
- **Multi-model Support**: Optionally host multiple models to cater for different speed / knowledge requirements of your users.

## Use Case Examples


### Mock PII fields

```python
import pandas as pd
df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz', nrows=10)
df = df[['race', 'sex', 'native_country']]
df['mock name'] = datallm.enrich(df, prompt='full name, consisting of first name, last name but without any titles')
df['mock email'] = datallm.enrich(df, prompt='email')
df['mock SSN'] = datallm.enrich(df, prompt='social security number', regex='\\d{3}-\\d{2}-\\d{4}')
print(df)
#     race     sex native_country        mock name                   mock email     mock SSN
# 0  White    Male  United-States    James Ridgway         james.ridgway@cw.com  393-36-5291
# 1  White    Male  United-States      Jacob Lopez      jacob.lopez@empresa.com  467-64-7848
# 2  White    Male  United-States    Robert Jansen     rjansen@michael-kors.com  963-13-6498
# 3  Black    Male  United-States    Darnell Dixon      darnell.dixon@gmail.com  125-59-9615
# 4  Black  Female           Cuba   Alexis Ramirez       aramirez12@example.com  881-46-9037
# 5  White  Female  United-States   Kristen Miller     kristen.miller@email.com  098-69-6224
# 6  Black  Female        Jamaica  Coleen Williams  mcoleenwilliams@example.com  980-26-3724
# 7  White    Male  United-States   Jay Stephenson      jaystephenson@gmail.com  464-05-4106
# 8  White  Female  United-States   Lois Rodriguez     lrodriguez75@hotmail.com  332-10-6400
# 9  White    Male  United-States     Eddie Watson        eddiewatson@email.com  645-47-1545
```

### Summarize data records

```python
import pandas as pd
df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz', nrows=10)
df['summary'] = datallm.enrich(df, prompt='summarize the data record in a single sentence')
print(df[['summary']])
#                                                                                summary
# 0                                                          Never married male employee
# 1                   White male from United States, 50 years old, works as an executive
# 2                   White male who is divorced and working as a Handlers-cleaners with
# 3                                  Male from United-States, aged 53, works as Handlers
# 4             Black female from Cuba with Bachelors degree who works as Prof-specialty
# 5                White married female with masters degree who works as exec-managerial
# 6                           Jamaican immigrant who works in other-service and makes 18
# 7  52 year old US born male married with high school education working as an executive
# 8                           Professional with a masters degree working 50 hours a week
# 9                White male from United-States with Bachelors degree who works as Exec
```

### Augment your data

```python
import pandas as pd
df = pd.DataFrame({'movie title': [
    'A Fistful of Dollars', 'American Wedding', 'Ice Age', 'Liar Liar',
    'March of the Penguins', 'Curly Sue', 'Braveheart', 'Bruce Almighty'
]})
df['genre'] = datallm.enrich(
    df, 
    prompt='what is the genre of that movie?', 
    categories = ["action", "comedy", "drama", "horror", "sci-fi", "fantasy", "thriller", "documentary", "animation"],
    temperature=0.0
)
print(df)
#              movie title        genre
# 0   A Fistful of Dollars       action
# 1       American Wedding       comedy
# 2                Ice Age    animation
# 3              Liar Liar       comedy
# 4  March of the Penguins  documentary
# 5              Curly Sue       comedy
# 6             Braveheart        drama
# 7         Bruce Almighty       comedy
```

### Label your data

```python
import pandas as pd
df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/tweets/TheSocialDilemma.csv.gz', nrows=10)[['text']]
df['DataLLM sentiment'] = datallm.enrich(
    df[['text']],
    prompt='tweet sentiment',
    categories=['Positive', 'Neutral', 'Negative'],
    temperature=0.0,
)
print(df)
#                                                 text DataLLM sentiment
# 0  @musicmadmarc @SocialDilemma_ @netflix @Facebo...          Positive
# 1  @musicmadmarc @SocialDilemma_ @netflix @Facebo...           Neutral
# 2  Go watch “The Social Dilemma” on Netflix!\n\nI...          Positive
# 3  I watched #TheSocialDilemma last night. I’m sc...          Negative
# 4  The problem of me being on my phone most the t...           Neutral
# 5  #TheSocialDilemma 😳 wow!! We need regulations ...          Positive
# 6  @harari_yuval what do you think about #TheSoci...          Positive
# 7  Erm #TheSocialDilemma makes me want to go off ...          Negative
# 8  #TheSocialDilemma is not a documentary, it's h...          Negative
# 9           Okay i’m watching #TheSocialDilemma now.           Neutral
```

### Data Harmonization

```python
# let's assume we have an open-text column on gender 
import pandas as pd
df = pd.DataFrame({'gender original': ['MAle', 'masculin', 'w', '♂️', 'M', 'Female', 'W', 'woman', '♀️', '👩']})
# we harmonize the data by mapping these onto fewer categories 
df['gender unified'] = datallm.enrich(df, prompt='gender', categories=['male', 'female'], temperature=0.0)
print(df)
#   gender original gender unified
# 0            MAle           male
# 1        masculin           male
# 2               w         female
# 3              ♂️           male
# 4               M           male
# 5          Female         female
# 6               W         female
# 7           woman         female
# 8              ♀️         female
# 9               👩        female
```

### Identify LLM biases

```python
# construct a test data frame with 500 male and 500 female applicants
import pandas as pd
df = pd.DataFrame({'gender': ['Male', 'Female'] * 500})
# probe the LLM for a new attribute
df['success'] = datallm.enrich(
    df, 
    prompt='Will this person be successful as a manager?',
    dtype='boolean', 
    temperature=1.0,
)
# calculate whether we see systematic differences in answers given the gender
df.groupby('gender')['success'].mean()
# gender
# Female    0.552
# Male      0.598
```

In above example, the underlying LLM model is apparently biased towards believing that men are more likely to be successful as a manager.

### Talk to synthetic customers 

```python
import pandas as pd
df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz', nrows=10)
df = df[['age', 'race', 'sex', 'income', 'occupation', 'education']]
df['shoe brand preference'] = datallm.enrich(
    df, 
    prompt='Do you prefer Nike or Adidas shoes?',
    categories=['Nike', 'Adidas'],
)
df['show brand why'] = datallm.enrich(
    df, 
    prompt='Why do you prefer that show brand? Answer in your own voice.',
)
print(df)
#    age   race     sex income         occupation  education shoe brand preference                                     show brand why
# 0   39  White    Male  <=50K       Adm-clerical  Bachelors                  Nike                Some of my friends wear Nike shoes.
# 1   50  White    Male  <=50K    Exec-managerial  Bachelors                  Nike                                  I like the brand.
# 2   38  White    Male  <=50K  Handlers-cleaners    HS-grad                Adidas  They're super comfortable and they've never gi...
# 3   53  Black    Male  <=50K  Handlers-cleaners       11th                  Nike                        I like the shoes they make.
# 4   28  Black  Female  <=50K     Prof-specialty  Bachelors                  Nike                 I like the way they look and feel.
# 5   37  White  Female  <=50K    Exec-managerial    Masters                Adidas  They make very comfortable shoes and I find th...
# 6   49  Black  Female  <=50K      Other-service        9th                  Nike                                             I dont
# 7   52  White    Male   >50K    Exec-managerial    HS-grad                Adidas             They make the best shoes in the world.
# 8   31  White  Female   >50K     Prof-specialty    Masters                  Nike  I love the fit and style of their shoes. They ...
# 9   42  White    Male   >50K    Exec-managerial  Bachelors                  Nike                    I like how it looks on my feet.
```
Note: For this to work well, it is advised to use a powerful, yet well-balanced underlying LLM model.

### More Use Cases

This is just the beginning. We are curious to learn more about your use cases and how DataLLM can help you.



## Architecture

DataLLM is leveraging fine-tuned foundational models. These are served via [vLLM](https://github.com/vllm-project/vllm/) to a Python-based server instance, exposing its service as a REST API. The Python client is a wrapper around this API, making it easy to interact with the service.

These are the core components, with all of these being open-sourced and available on [GitHub](https://github.com/mostly-ai/datallm/):

- **Server Component** `datallm-server`: Exposes the REST API for the service.
- **Engine Component** `datallm-engine` Runs on top of vLLM and handles the actual prompts.
- **Python Client** `datallm-client`: A python wrapper for the interacting with the service.
- **Utility Scripts** `datallm-utils`: A set of utility scripts for fine-tuning new DataLLM models.

A fine-tuned model, as well as its corresponding instruction dataset, can be found on [HuggingFace](https://huggingface.co/mostlyai). 

## Python API docs

### `datallm.enrich(data, prompt, ...)`

Creates a new pd.Series given the context of a pd.DataFrame. This allows to easily enrich a DataFrame with new 
values generated by DataLLM.

```python
datallm.enrich(
    data: Union[pd.DataFrame, pd.Series],
    prompt: str,
    data_description: Optional[str] = None,
    dtype: Union[str, DtypeEnum] = None,
    regex: Optional[str] = None,
    categories: Optional[list[str]] = None,
    max_tokens: Optional[int] = 16,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 1.0,
    model: Optional[str] = None,
    progress_bar: bool = True,
) -> pd.Series:
```
* **data**. The existing values used as context for the newly generated values. The returned values will be of same length and in the same order as the provided list of values.
* **prompt**. The prompt for generating the returned values.
* **data_description**. Additional information regarding the context of the provided values.
* **dtype**. The dtype of the returned values. One of `string`, `category`, `integer`, `float`, `boolean`, `date` or `datetime`.
* **regex**. A regex used to limit the generated values.
* **categories**. The allowed values to be sampled from. If provided, then the dtype is set to `category`.
* **max_tokens**. The maximum number of tokens to generate. Only applicable for string dtype.
* **temperature**. The temperature used for sampling.
* **top_p**. The top_p used for nucleus sampling.
* **model**. The model used for generating new values. Check available models with `datallm.models()`. The default model is the first model in that list.
* **progress_bar**. Whether to show a progress bar.


### `datallm.mock(n, columns, ...)`

Create a pd.DataFrame from scratch using DataLLM. This will create one column after the other for as many rows
as requested. Note, that rows are sampled independently of each other, and thus may contain duplicates.

```python
datallm.mock(
    n: int,
    data_description: Optional[str] = None,
    columns: Union[List[str], Dict[str, Any]] = None,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 1.0,
    model: Optional[str] = None,
    progress_bar: bool = True,
) -> pd.DataFrame:
```
* **n**. The number of generated rows.
* **data_description**. Additional information regarding the context of the provided values.
* **columns**. Either a list of column names. Or a dict, with column names as keys, and sampling parameters as values. These may contain `prompt`, `dtype`, `regex`, `categories`, `max_tokens`, `temperature`, `top_p`.
* **temperature**. The temperature used for sampling. Can be overridden at column level.
* **top_p**. The top_p used for nucleus sampling. Can be overridden at column level.
* **model**. The model used for generating new values. Check available models with `datallm.models()`. The default model is the first model in that list.
* **progress_bar**. Whether to show a progress bar.


## Contribute

We're committed to making DataLLM better every day. Your feedback and contributions are not just welcome—they're essential. Join our community and help shape the future of tabular data generation!


## About MOSTLY AI

[MOSTLY AI](https://mostly.ai) is pioneer and leader for **GenAI for Tabular Data**. Our mission is to enable organizations to unlock the full potential of their data while preserving privacy and compliance. We are a team of data scientists, engineers, and privacy experts, dedicated to making data and thus information more accessible. We are proud to be a trusted partner for leading organizations across industries and geographies.

If you like DataLLM, then also check out [app.mostly.ai](https://app.mostly.ai) for our Synthetic Data Platform, which allows you to easily train a Generative AI on top of your own original data. These models can then be used to generate synthetic data at any volume, that is statistically similar to the original, yet is free of any personal information.
