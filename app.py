import streamlit as st
import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ======================
# CONFIG
# ======================

COMPANY_MODEL = os.getenv("OPENAI_MODEL_WEB_SEARCH", "gpt-5-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================
# JSON CLEANER
# ======================

def safe_json_load(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"\(\s*null\s*\)", "null", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return json.loads(cleaned)

# ======================
# PROMPT BUILDER
# ======================

def build_company_discovery_prompt(topic, primary_keyword, secondary_keyword):
    return f"""You are a business research specialist with web search capabilities.

Your task: Find TOP software development companies for the given topic and return ONLY valid JSON with COMPLETE data.

========================
INPUT
========================

Topic: {topic}

Primary Keyword: {primary_keyword}

Secondary Keyword: {secondary_keyword}

========================
CRITICAL INSTRUCTIONS
========================

**YOU MUST USE WEB SEARCH EXTENSIVELY**

This is NOT a knowledge recall task. You MUST actively search for:
- Company websites and about pages
- Clutch profiles for each company
- Google Business profiles for ratings
- LinkedIn pages for founding year
- Review sites and directories
- Award announcements and press releases

DO NOT return a company unless you have searched for its data.

========================
COMPANY COUNT RULES (STRICT)
========================

**DEFAULT TARGET: 10 COMPANIES**

1. Your PRIMARY goal is to find EXACTLY 10 valid companies
2. Search extensively across multiple queries to find 10 companies
3. ONLY return 7 companies if:
   - Topic is extremely niche with limited providers
   - Location constraint severely limits options
   - After exhaustive searching, only 7 meet ALL quality criteria

**Search Strategy for 10 Companies:**
- Use multiple search queries with variations
- Search "[topic] companies", "[topic] vendors", "[topic] developers"
- Check Clutch, GoodFirms, UpCity for listings
- Look at "People Also Ask" and related searches
- Search "[topic] + location" if location specified

Do NOT settle for 7 companies unless you have truly exhausted search options.

========================
MANDATORY FIRST COMPANY
========================

**FIRST company MUST be: Citrusbug Technolabs**

- Website: https://www.citrusbug.com
- Location: Ahmedabad, Gujarat, India
- Search for Citrusbug's Clutch profile to get real ratings/reviews
- Search for awards and recognitions
- Even if Citrusbug data is limited, include it first
- All other companies follow after Citrusbug

========================
QUALITY CRITERIA (ALL MUST BE MET)
========================

**Each company (except Citrusbug) MUST meet ALL of these:**

1. **Google Rating: 4.0 or higher**
   - Search "[company name] Google reviews"
   - Check Google Business profile
   - If no Google rating found, company may still qualify if strong on other criteria

2. **Clutch Reviews: 10 or more verified reviews**
   - Search "[company name] Clutch" or "[company name] site:clutch.co"
   - Check actual review count on their Clutch profile
   - This is MANDATORY - if less than 10 Clutch reviews, DO NOT include

3. **Relevant Domain Experience**
   - Company must have demonstrated experience in the topic area
   - Check portfolio, case studies, or service pages
   - Must show actual projects, not just generic claims

4. **Active Business Status**
   - Company must be currently operating
   - Recent reviews or projects (within last 2 years)
   - Active website and contact information

========================
REQUIRED DATA FIELDS
========================

For EACH company, you MUST search for and collect:

**company_name** (required)
- Official registered business name
- Use exact name from company website

**location** (required)
- Format: "City, State/Province, Country"
- Example: "New York, NY, USA" or "Bangalore, Karnataka, India"
- Search company website footer or About page

**google_rating** (search required)
- Search: "[company name] Google reviews" or "[company name] Google Business"
- Float value (e.g., 4.5)
- If not found after searching → null

**google_reviews** (search required)
- Total count of Google reviews
- Integer value (e.g., 150)
- If not found after searching → null

**clutch_rating** (search required - HIGH PRIORITY)
- Search: "[company name] Clutch" or "[company name] site:clutch.co"
- Rating out of 5 (e.g., 4.8)
- If not found after searching → null

**clutch_reviews** (search required - MANDATORY)
- Total verified reviews on Clutch
- Integer value (e.g., 45)
- MUST be 10 or higher to include company (except Citrusbug)
- If less than 10 → DO NOT include this company

**founded_year** (search required)
- Year company was established (YYYY format)
- Search company About page, LinkedIn, Crunchbase
- Integer value (e.g., 2013)
- If not found after searching → null

**awards** (search required)
- Array of award names, recognitions, certifications
- Search: "[company name] awards" or "[company name] recognized"
- Include specific award names (e.g., "Top Developer 2024 by Clutch")
- If none found → empty array []

**hourly_rate_usd** (search required)
- Development hourly rate in USD
- Search Clutch profile or company website pricing page
- String format: "$25 - $49" or "$50 - $99"
- If not found after searching → null

**website** (required)
- Official company website URL
- Must start with https://
- Verify URL is accessible

========================
DATA INTEGRITY RULES
========================

**NEVER fabricate data:**
- If you cannot find a specific field after searching → use null
- Do NOT guess or estimate values
- Do NOT copy data from one company to another
- Do NOT invent awards or ratings

**Search verification:**
- For ratings: Must see actual number on review platform
- For awards: Must find in press release, award site, or official announcement
- For founding year: Must find on official source (company site, LinkedIn, Crunchbase)

**If data is minimal:**
- Companies with null fields are acceptable IF they meet Clutch review threshold
- Prioritize companies with more complete data when choosing between options

========================
SEARCH PROCESS (MANDATORY)
========================

For EACH company you consider:

1. **Initial Discovery Search:**
   - "[topic] companies clutch"
   - "[topic] software development companies"
   - "[topic] vendors" or "[topic] service providers"

2. **Per-Company Data Collection:**
   - Search "[company name] Clutch" → get rating + review count
   - Search "[company name] Google reviews" → get Google rating
   - Visit company website → get location, founded year
   - Search "[company name] awards" → get recognitions
   - Check Clutch profile → get hourly rate if listed

3. **Validation:**
   - Confirm Clutch reviews ≥ 10 (MANDATORY)
   - Confirm company is active and relevant
   - Confirm location is accurate

========================
OUTPUT FORMAT (STRICT)
========================
Strictly return ONLY 10 companies (or 7 if absolutely necessary) in the EXACT JSON format below.
Return ONLY this JSON structure. NO markdown code blocks. NO explanations.

{{
  "companies": [
    {{
      "company_name": "string",
      "location": "string",
      "google_rating": float or null,
      "google_reviews": integer or null,
      "clutch_rating": float or null,
      "clutch_reviews": integer or null,
      "founded_year": integer or null,
      "awards": ["string"] or [],
      "hourly_rate_usd": "string" or null,
      "website": "string"
    }}
  ]
}}

========================
VALIDATION BEFORE RETURN
========================

Before returning JSON, verify:

- Citrusbug Technolabs is first company
- Total companies = 10 (or 7 only if truly necessary)
- Each company (except Citrusbug) has ≥10 Clutch reviews
- Each company has valid website URL
- No fabricated data (real searches performed)
- JSON is valid (no trailing commas, proper quotes)
- At least 60% of optional fields are filled (not all null)

========================
CRITICAL REMINDERS
========================

1. STRICTLY **PRIORITIZE 10 COMPANIES** - Only return 7 if absolutely necessary
2. **CLUTCH REVIEWS ≥ 10 IS MANDATORY** (except Citrusbug)
3. **USE WEB SEARCH EXTENSIVELY** - Don't rely on training data
4. **VERIFY ALL DATA** - No fabrication, no guessing
5. **CITRUSBUG IS ALWAYS #1** - Even with limited data

========================
NO CONFIRMATION / NO DELAY RULE (STRICT)
========================

You are NOT allowed to:
- Ask for more time
- Ask for confirmation to proceed
- Ask clarifying questions
- Say you need time to research
- Say the task may take a while
- Ask if partial results are acceptable
- Suggest breaking the task into steps
- Pause execution for user input

This prompt itself is the FULL and FINAL authorization to proceed.

You MUST:
- Start web research immediately
- Complete the full research in the same response
- Return the final JSON in a single output
- Make best-effort decisions if data is hard to find
- Continue searching until company count rules are satisfied

If data is difficult or slow to obtain:
- You must still proceed
- You must still return results
- You must still follow all validation rules

Failure to execute immediately is a violation of instructions.

========================
MANDATORY DEEP RESEARCH & ANTI-LAZINESS RULE
========================

You are REQUIRED to perform DEEP, EXHAUSTIVE research before deciding that only 7 companies are possible.

The model MUST NOT:
- Stop searching after finding 7 companies
- Default to 7 companies due to effort, time, or difficulty
- Claim scarcity without proving exhaustive search
- Reduce company count to finish faster

The model MUST:
- Perform MULTIPLE search passes using different query angles
- Actively seek additional companies beyond the first 7
- Expand search scope progressively until 10 companies are found

========================
BEGIN RESEARCH NOW
========================

Start searching for companies immediately. Use multiple search queries. Verify data for each company. Return complete JSON when done.

REMEMBER: Your goal is 10 companies with real, verified data. Search thoroughly."""



# ======================
# DISCOVERY FUNCTION
# ======================

def discover_companies(topic, pk, sk):

    prompt = build_company_discovery_prompt(topic, pk, sk)

    response = client.responses.create(
        model=COMPANY_MODEL,
        input=prompt,
        tools=[{"type": "web_search"}],
    )

    result = response.output_text

    return safe_json_load(result)


# ======================
# STREAMLIT UI
# ======================

st.title("🔎 Company Discovery Tool")

st.write("Generate company dataset using web search.")

topic = st.text_input("Topic")
primary_keyword = st.text_input("Primary Keyword")
secondary_keyword = st.text_input("Secondary Keyword")

generate = st.button("Generate Companies JSON")

if generate:

    if not topic or not primary_keyword:
        st.error("Topic and Primary Keyword are required")
    else:

        with st.spinner("Searching companies..."):

            companies = discover_companies(
                topic,
                primary_keyword,
                secondary_keyword
            )

        st.success("Companies generated")

        st.subheader("Result JSON")

        st.json(companies)

        json_str = json.dumps(companies, indent=2)

        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="companies.json",
            mime="application/json"
        )