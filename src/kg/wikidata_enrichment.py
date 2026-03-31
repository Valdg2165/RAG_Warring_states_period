
import re
import time
import requests
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

WS   = Namespace("http://warring-states.kg/ontology#")
WSI  = Namespace("http://warring-states.kg/instance/")
PROV = Namespace("http://www.w3.org/ns/prov#")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "WarringStatesKG/1.0 (educational; contact: student@esilv.fr)"}


def _parse_year(date_str: str) -> str:
    if not date_str:
        return ""
    if date_str.startswith("-"):
        return date_str[:5]   # "-YYYY"
    return date_str[:4]       # "YYYY"

KG_FILE       = "kg_artifacts/warring_states.ttl"
EXPANDED_FILE = "kg_artifacts/warring_states_expanded.ttl"

PERSON_QIDS = {
    "Confucius":    "Q4604",
    "Mencius":      "Q188903",
    "Shang Yang":   "Q351345",
    "Sun Tzu":      "Q37151",
    "Han Fei":      "Q28959",
    "Li Si":        "Q152919",
    "Qin Shi Huang":"Q7192",
    "Wu Qi":        "Q698899",
    "Xunzi":        "Q216072",
    "Laozi":        "Q9333",
    "Mozi":         "Q272411",
    "Sima Qian":    "Q9372",
    "Bai Qi":       "Q701746",
    "Wang Jian":    "Q706855",
    "Zhuangzi":     "Q272411",  # approx
    "Fan Sui":      "Q45384011",
    "Su Qin":       "Q45503599",
}

SCHOOL_QIDS = {
    "Confucianism":            "Q9581",
    "Legalism":                "Q720866",
    "Mohism":                  "Q720866",
    "Hundred Schools of Thought": "Q864947",
}

BATTLE_QIDS = {
    "Battle of Changping": "Q1195760",
    "Battle of Yique":     "Q1291237",
    "Battle of Maling":    "Q2005185",
}

STATE_QIDS = {
    "Qin":  "Q207788",   # State of Qin
    "Chu":  "Q163354",   # State of Chu
    "Wei":  "Q208010",   # State of Wei (Warring States)
    "Zhao": "Q207840",   # State of Zhao
    "Han":  "Q1053796",  # State of Han
    "Yan":  "Q207809",   # State of Yan
    "Qi":   "Q207804",   # State of Qi
    "Zhou": "Q35216",    # Zhou dynasty
    "Jin":  "Q207862",   # State of Jin
    "Lu":   "Q207814",   # State of Lu
    "Wu":   "Q207807",   # State of Wu
    "Yue":  "Q207861",   # State of Yue
}

WARRING_STATES_QID = "Q185063"



def sparql_query(query: str) -> list[dict]:
    try:
        r = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    except Exception as e:
        print(f"    [SPARQL error] {e}")
        return []


def uri_safe(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def ensure_entity(g: Graph, label: str, cls: URIRef = None) -> URIRef:
    uri = WSI[uri_safe(label)]
    g.add((uri, RDFS.label, Literal(label)))
    if cls:
        g.add((uri, RDF.type, cls))
    return uri


PROV_SOURCE = URIRef("https://www.wikidata.org/")


def add_triple(g: Graph, s_label: str, pred: URIRef,
               o_val, is_literal=False,
               s_cls=None, o_cls=None) -> bool:
    s = ensure_entity(g, s_label, s_cls)
    o = o_val if is_literal else ensure_entity(g, o_val, o_cls)
    t = (s, pred, o)
    if t not in g:
        g.add(t)
        g.add((s, PROV.wasDerivedFrom, PROV_SOURCE))
        return True
    return False


NEW_PROPS = [
    (WS.hasCapital,       "has capital"),
    (WS.locatedIn,        "located in"),
    (WS.bornIn,           "born in"),
    (WS.diedIn,           "died in"),
    (WS.birthYear,        "birth year"),
    (WS.deathYear,        "death year"),
    (WS.foundedYear,      "founded year"),
    (WS.endYear,          "end year"),
    (WS.hasFather,        "has father"),
    (WS.hasChild,         "has child"),
    (WS.heldPosition,     "held position"),
    (WS.foughtIn,         "fought in"),
    (WS.hasParticipant,   "has participant"),
    (WS.wonBy,            "won by"),
    (WS.partOf,           "part of"),
    (WS.follows,          "follows"),
    (WS.followedBy,       "followed by"),
    (WS.influencedBy,     "influenced by"),
    (WS.studentOf,        "student of"),
    (WS.hasStudent,       "has student"),
    (WS.memberOf,         "member of"),
    (WS.fieldOfWork,      "field of work"),
    (WS.movement,         "movement"),
    (WS.servedIn,         "served in"),
    (WS.headOfState,      "head of state"),
    (WS.replaces,         "replaces"),
    (WS.replacedBy,       "replaced by"),
    (WS.instanceOf,       "instance of"),
]



def fetch_person_details(qids_dict: dict) -> list[dict]:
    """Birth/death, teacher, influenced-by for known persons."""
    values = " ".join(f"wd:{q}" for q in qids_dict.values())
    q = f"""
    SELECT DISTINCT ?person ?personLabel ?birth ?death
                    ?teacher ?teacherLabel ?inf ?infLabel
                    ?role ?roleLabel ?bornIn ?bornInLabel WHERE {{
      VALUES ?person {{ {values} }}
      OPTIONAL {{ ?person wdt:P569 ?birth . }}
      OPTIONAL {{ ?person wdt:P570 ?death . }}
      OPTIONAL {{ ?person wdt:P1066 ?teacher . }}
      OPTIONAL {{ ?person wdt:P737 ?inf . }}
      OPTIONAL {{ ?person wdt:P106 ?role . }}
      OPTIONAL {{ ?person wdt:P19 ?bornIn . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    return sparql_query(q)


def fetch_state_details(qids_dict: dict) -> list[dict]:
    values = " ".join(f"wd:{q}" for q in qids_dict.values())
    q = f"""
    SELECT DISTINCT ?state ?stateLabel ?capital ?capitalLabel
                    ?ruler ?rulerLabel ?inception ?end
                    ?successor ?successorLabel ?predecessor ?predecessorLabel WHERE {{
      VALUES ?state {{ {values} }}
      OPTIONAL {{ ?state wdt:P36 ?capital . }}
      OPTIONAL {{ ?state wdt:P6|wdt:P35 ?ruler . }}
      OPTIONAL {{ ?state wdt:P571 ?inception . }}
      OPTIONAL {{ ?state wdt:P576 ?end . }}
      OPTIONAL {{ ?state wdt:P1366 ?successor . }}
      OPTIONAL {{ ?state wdt:P1365 ?predecessor . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    return sparql_query(q)


def fetch_battle_details(qids_dict: dict) -> list[dict]:
    values = " ".join(f"wd:{q}" for q in qids_dict.values())
    q = f"""
    SELECT DISTINCT ?battle ?battleLabel ?participant ?participantLabel
                    ?winner ?winnerLabel ?date ?location ?locationLabel WHERE {{
      VALUES ?battle {{ {values} }}
      OPTIONAL {{ ?battle wdt:P710 ?participant . }}
      OPTIONAL {{ ?battle wdt:P1111 ?winner . }}
      OPTIONAL {{ ?battle wdt:P585 ?date . }}
      OPTIONAL {{ ?battle wdt:P276 ?location . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    return sparql_query(q)


def fetch_warring_states_people() -> list[dict]:
    q = f"""
    SELECT DISTINCT ?person ?personLabel ?birth ?death ?role ?roleLabel WHERE {{
      ?person wdt:P31 wd:Q5 .
      {{ ?person wdt:P607/wdt:P361 wd:{WARRING_STATES_QID} . }}
      UNION
      {{ ?person wdt:P19/wdt:P361 wd:{WARRING_STATES_QID} . }}
      OPTIONAL {{ ?person wdt:P569 ?birth . }}
      OPTIONAL {{ ?person wdt:P570 ?death . }}
      OPTIONAL {{ ?person wdt:P106 ?role . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 200
    """
    return sparql_query(q)


def fetch_philosophical_network() -> list[dict]:
    q = f"""
    SELECT DISTINCT ?person ?personLabel ?teacher ?teacherLabel ?inf ?infLabel WHERE {{
      ?person wdt:P31 wd:Q5 .
      ?person wdt:P569 ?birth .
      FILTER(?birth > "-600-01-01"^^xsd:dateTime && ?birth < "-200-01-01"^^xsd:dateTime)
      OPTIONAL {{ ?person wdt:P1066 ?teacher . }}
      OPTIONAL {{ ?person wdt:P737 ?inf . }}
      FILTER(BOUND(?teacher) || BOUND(?inf))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 300
    """
    return sparql_query(q)


def fetch_battles_of_period() -> list[dict]:
    q = f"""
    SELECT DISTINCT ?battle ?battleLabel ?participant ?participantLabel
                    ?winner ?winnerLabel ?date ?location ?locationLabel WHERE {{
      ?battle wdt:P31/wdt:P279* wd:Q178561 .
      ?battle wdt:P361 wd:{WARRING_STATES_QID} .
      OPTIONAL {{ ?battle wdt:P710 ?participant . }}
      OPTIONAL {{ ?battle wdt:P1111 ?winner . }}
      OPTIONAL {{ ?battle wdt:P585 ?date . }}
      OPTIONAL {{ ?battle wdt:P276 ?location . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 200
    """
    return sparql_query(q)


def fetch_state_battles() -> list[dict]:
    values = " ".join(f"wd:{q}" for q in STATE_QIDS.values())
    q = f"""
    SELECT DISTINCT ?battle ?battleLabel ?participant ?participantLabel
                    ?winner ?winnerLabel ?date WHERE {{
      VALUES ?participant {{ {values} }}
      ?battle wdt:P710 ?participant .
      ?battle wdt:P31/wdt:P279* wd:Q178561 .
      OPTIONAL {{ ?battle wdt:P1111 ?winner . }}
      OPTIONAL {{ ?battle wdt:P585 ?date . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 200
    """
    return sparql_query(q)


def fetch_notable_works() -> list[dict]:
    values = " ".join(f"wd:{q}" for q in {**PERSON_QIDS}.values())
    q = f"""
    SELECT DISTINCT ?author ?authorLabel ?work ?workLabel WHERE {{
      VALUES ?author {{ {values} }}
      ?work wdt:P50 ?author .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 100
    """
    return sparql_query(q)



def run():
    g = Graph()
    g.parse(KG_FILE, format="turtle")
    g.bind("ws", WS); g.bind("wsi", WSI); g.bind("prov", PROV)
    before = len(g)
    print(f"Loaded KG: {before:,} triples\n")

    for prop, label in NEW_PROPS:
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.label, Literal(label)))

    qid_to_label = {}
    for d in [PERSON_QIDS, STATE_QIDS, BATTLE_QIDS, SCHOOL_QIDS]:
        for lbl, qid in d.items():
            qid_to_label[f"http://www.wikidata.org/entity/{qid}"] = lbl

    n = 0 

    print("[1/6] Person birth/death/teacher/influenced...")
    rows = fetch_person_details(PERSON_QIDS)
    for row in rows:
        p_uri = row.get("person", {}).get("value", "")
        p_lbl = qid_to_label.get(p_uri) or row.get("personLabel", {}).get("value", "")
        if not p_lbl:
            continue

        birth = _parse_year(row.get("birth", {}).get("value", ""))
        death = _parse_year(row.get("death", {}).get("value", ""))
        teacher_lbl = row.get("teacherLabel", {}).get("value", "")
        inf_lbl     = row.get("infLabel",     {}).get("value", "")
        role_lbl    = row.get("roleLabel",    {}).get("value", "")
        born_in_lbl = row.get("bornInLabel",  {}).get("value", "")

        if birth and birth.lstrip("-").isdigit():
            n += add_triple(g, p_lbl, WS.birthYear,
                            Literal(birth, datatype=XSD.string), is_literal=True,
                            s_cls=WS.Person)
        if death and death.lstrip("-").isdigit():
            n += add_triple(g, p_lbl, WS.deathYear,
                            Literal(death, datatype=XSD.string), is_literal=True,
                            s_cls=WS.Person)
        if teacher_lbl and teacher_lbl != p_lbl:
            n += add_triple(g, p_lbl, WS.studentOf, teacher_lbl,
                            s_cls=WS.Person, o_cls=WS.Person)
        if inf_lbl and inf_lbl != p_lbl:
            n += add_triple(g, p_lbl, WS.influencedBy, inf_lbl,
                            s_cls=WS.Person, o_cls=WS.Person)
        if role_lbl:
            n += add_triple(g, p_lbl, WS.heldPosition, role_lbl,
                            s_cls=WS.Person)
        if born_in_lbl:
            n += add_triple(g, p_lbl, WS.bornIn, born_in_lbl,
                            s_cls=WS.Person, o_cls=WS.Place)

    print(f"  +{n} (total so far)")
    time.sleep(1)

    print("[2/6] State capitals, rulers, dates...")
    prev = n
    rows = fetch_state_details(STATE_QIDS)
    for row in rows:
        s_uri = row.get("state", {}).get("value", "")
        s_lbl = qid_to_label.get(s_uri) or row.get("stateLabel", {}).get("value", "")
        if not s_lbl:
            continue

        cap   = row.get("capitalLabel",     {}).get("value", "")
        ruler = row.get("rulerLabel",       {}).get("value", "")
        succ  = row.get("successorLabel",   {}).get("value", "")
        pred  = row.get("predecessorLabel", {}).get("value", "")
        inc   = _parse_year(row.get("inception", {}).get("value", ""))
        end   = _parse_year(row.get("end",       {}).get("value", ""))

        if cap:
            n += add_triple(g, s_lbl, WS.hasCapital, cap,
                            s_cls=WS.State, o_cls=WS.Place)
        if ruler:
            n += add_triple(g, s_lbl, WS.headOfState, ruler,
                            s_cls=WS.State, o_cls=WS.Person)
        if succ:
            n += add_triple(g, s_lbl, WS.followedBy, succ,
                            s_cls=WS.State)
        if pred:
            n += add_triple(g, s_lbl, WS.follows, pred,
                            s_cls=WS.State)
        if inc and inc.lstrip("-").isdigit():
            n += add_triple(g, s_lbl, WS.foundedYear,
                            Literal(inc, datatype=XSD.string), is_literal=True,
                            s_cls=WS.State)
        if end and end.lstrip("-").isdigit():
            n += add_triple(g, s_lbl, WS.endYear,
                            Literal(end, datatype=XSD.string), is_literal=True,
                            s_cls=WS.State)

    print(f"  +{n - prev} (total {n})")
    time.sleep(1)

    print("[3/6] Battle participants & outcomes (known battles)...")
    prev = n
    rows = fetch_battle_details(BATTLE_QIDS)
    for row in rows:
        b_uri = row.get("battle",      {}).get("value", "")
        b_lbl = qid_to_label.get(b_uri) or row.get("battleLabel", {}).get("value", "")
        p_lbl = row.get("participantLabel", {}).get("value", "")
        w_lbl = row.get("winnerLabel",      {}).get("value", "")
        loc   = row.get("locationLabel",    {}).get("value", "")
        date  = _parse_year(row.get("date", {}).get("value", ""))

        if b_lbl and p_lbl:
            n += add_triple(g, b_lbl, WS.hasParticipant, p_lbl,
                            s_cls=WS.Battle)
        if b_lbl and w_lbl:
            n += add_triple(g, b_lbl, WS.wonBy, w_lbl,
                            s_cls=WS.Battle)
        if b_lbl and loc:
            n += add_triple(g, b_lbl, WS.locatedIn, loc,
                            s_cls=WS.Battle, o_cls=WS.Place)
        if b_lbl and date and date.lstrip("-").isdigit():
            n += add_triple(g, b_lbl, WS.foundedYear,
                            Literal(date, datatype=XSD.string), is_literal=True,
                            s_cls=WS.Battle)
    print(f"  +{n - prev} (total {n})")
    time.sleep(1)

    print("[4/6] All battles involving known states...")
    prev = n
    rows = fetch_state_battles()
    for row in rows:
        b_lbl = row.get("battleLabel",      {}).get("value", "")
        p_uri = row.get("participant",      {}).get("value", "")
        p_lbl = qid_to_label.get(p_uri) or row.get("participantLabel", {}).get("value", "")
        w_lbl = row.get("winnerLabel",      {}).get("value", "")
        date  = _parse_year(row.get("date", {}).get("value", ""))

        if b_lbl and p_lbl:
            n += add_triple(g, b_lbl, WS.hasParticipant, p_lbl,
                            s_cls=WS.Battle, o_cls=WS.State)
        if b_lbl and w_lbl:
            n += add_triple(g, b_lbl, WS.wonBy, w_lbl,
                            s_cls=WS.Battle)
        if b_lbl and date and date.lstrip("-").isdigit():
            n += add_triple(g, b_lbl, WS.foundedYear,
                            Literal(date, datatype=XSD.string), is_literal=True,
                            s_cls=WS.Battle)
    print(f"  +{n - prev} (total {n})")
    time.sleep(1)

    print("[5/6] Philosophical teacher/influence network...")
    prev = n
    rows = fetch_philosophical_network()
    for row in rows:
        p_lbl = row.get("personLabel",  {}).get("value", "")
        t_lbl = row.get("teacherLabel", {}).get("value", "")
        i_lbl = row.get("infLabel",     {}).get("value", "")
        if p_lbl and t_lbl and p_lbl != t_lbl:
            n += add_triple(g, p_lbl, WS.studentOf, t_lbl,
                            s_cls=WS.Person, o_cls=WS.Person)
        if p_lbl and i_lbl and p_lbl != i_lbl:
            n += add_triple(g, p_lbl, WS.influencedBy, i_lbl,
                            s_cls=WS.Person, o_cls=WS.Person)
    print(f"  +{n - prev} (total {n})")
    time.sleep(1)

    print("[6/6] Notable works authored by key figures...")
    prev = n
    rows = fetch_notable_works()
    for row in rows:
        a_uri = row.get("author", {}).get("value", "")
        a_lbl = qid_to_label.get(a_uri) or row.get("authorLabel", {}).get("value", "")
        w_lbl = row.get("workLabel", {}).get("value", "")
        if a_lbl and w_lbl:
            n += add_triple(g, a_lbl, WS.authored, w_lbl, s_cls=WS.Person)
    print(f"  +{n - prev} (total {n})")

    after = len(g)
    g.serialize(destination=EXPANDED_FILE, format="turtle")
    print(f"\nExpanded KG saved -> {EXPANDED_FILE}")
    print(f"  Before : {before:,} triples")
    print(f"  After  : {after:,} triples  (+{after-before:,})")

    from collections import Counter
    type_counts = Counter(
        str(o).split("#")[-1]
        for _, p, o in g.triples((None, RDF.type, None))
        if "ontology" in str(o)
    )
    print("\nInstance types:")
    for t, c in type_counts.most_common():
        print(f"  {t:30s} {c}")


if __name__ == "__main__":
    run()