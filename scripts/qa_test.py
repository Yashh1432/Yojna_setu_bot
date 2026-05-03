import sys
import os
import time
import requests
import json
import uuid

BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:5000')
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

def get_new_phone():
    return f"qa_{uuid.uuid4().hex[:8]}"

def chat(phone, message):
    payload = {
        "phone_number": phone,
        "message": message,
        "input_type": "text"
    }
    try:
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=20)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def print_result(name, input_text, extracted, response, schemes, passed, reason=""):
    print(f"====================================")
    print(f"TEST CASE: {name}")
    print(f"INPUT: {input_text}")
    print(f"EXTRACTED: {extracted}")
    print(f"BOT RESPONSE: {response}")
    print(f"SCHEMES RETURNED: {json.dumps(schemes, indent=2, ensure_ascii=False)}")
    print(f"STATUS: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"REASON: {reason}")
    print(f"====================================\n")

def check_fail_conditions(response_data, expected_state=None, avoid_states=None):
    # Check for "Available for <user_state>" used incorrectly (like showing it when not applicable or raw text)
    # Check for raw benefits paragraph
    resp_text = response_data.get("response", "")
    schemes = response_data.get("schemes", [])
    
    if len(resp_text) > 300: # heuristic for raw dump
        return False, "Response too long, likely raw text dump"
        
    for s in schemes:
        if expected_state and s.get("state") and s.get("state").lower() != "all india" and s.get("state").lower() != expected_state.lower():
            return False, f"Scheme from wrong state appears: {s.get('state')}"
        if avoid_states and s.get("state") and s.get("state").lower() in [av.lower() for av in avoid_states]:
            return False, f"Scheme from wrong state appears: {s.get('state')}"
        if "State: Not specified" in str(s):
            return False, "'State: Not specified' appears in scheme card"
        if len(s.get("benefits_summary", "")) > 150:
             return False, "Raw benefits paragraph shown in chat"
            
    return True, ""

def test_1():
    name = "TEST 1: FULL FREE SPEECH INPUT"
    phone = get_new_phone()
    # First choose language to avoid language prompt issues
    chat(phone, "English")
    
    inp = "I am a farmer aged 48 from Karnataka, income 2 lakh, show me schemes"
    res = chat(phone, inp)
    
    # We can't directly see 'extracted' from response, we infer from what it asks next or schemes
    schemes = res.get("schemes", [])
    resp_text = res.get("response", "").lower()
    
    passed = True
    reason = ""
    
    if "farmer" in resp_text or "occupation" in resp_text:
        passed = False; reason = "Asked occupation again"
    
    # either asks missing field or shows schemes
    if not schemes and "?" not in resp_text:
        passed = False; reason = "Didn't ask missing field or show schemes"
        
    ok, err = check_fail_conditions(res, expected_state="Karnataka")
    if not ok:
        passed = False; reason = err
        
    extracted = "farmer, 48, Karnataka, 2 lakh (inferred)"
    print_result(name, inp, extracted, res.get("response"), schemes, passed, reason)

def test_2():
    name = "TEST 2: TYPOS + TRANSLITERATION"
    phone = get_new_phone()
    chat(phone, "Hindi")
    
    inp = "kisan hu 50 saal ka hun income 3 lakh from maharastra"
    res = chat(phone, inp)
    
    resp_text = res.get("response", "").lower()
    passed = True
    reason = ""
    
    if "maharashtra" not in resp_text and "schemes" not in resp_text and "?" not in resp_text:
        # Check if it asks for something already provided
        pass
    
    # check if any repeated questions
    if "occupation" in resp_text or "age" in resp_text or "state" in resp_text:
        passed = False; reason = "Asked repeated question despite input"
        
    ok, err = check_fail_conditions(res, expected_state="Maharashtra")
    if not ok:
        passed = False; reason = err
        
    extracted = "farmer, 50, 3 lakh, Maharashtra (inferred)"
    print_result(name, inp, extracted, res.get("response"), res.get("schemes", []), passed, reason)

def test_3():
    name = "TEST 3: NON-HINDI LANGUAGE (TAMIL)"
    phone = get_new_phone()
    
    inp = "நான் விவசாயி, வயது 45, வருமானம் 2 லட்சம், கர்நாடகா"
    res = chat(phone, inp)
    
    resp_text = res.get("response", "")
    passed = True
    reason = ""
    
    # Check if English is in response
    if any(c.isascii() and c.isalpha() for c in resp_text.replace("Karnataka", "").replace("English", "")):
        # some english letters might exist (like scheme names) but fallback to english shouldn't happen for the main prompt
        if "Please" in resp_text or "Select" in resp_text or "Sorry" in resp_text:
            passed = False; reason = "English leakage detected"
            
    ok, err = check_fail_conditions(res, expected_state="Karnataka")
    if not ok:
        passed = False; reason = err
        
    extracted = "farmer, 45, 2 lakh, Karnataka (inferred)"
    print_result(name, inp, extracted, resp_text, res.get("schemes", []), passed, reason)

def test_4():
    name = "TEST 4: NUMERIC INPUT BUG"
    phone = get_new_phone()
    chat(phone, "English")
    chat(phone, "I am a student from Delhi") # to trigger "what is your percentage/income"
    
    inp = "2"
    res = chat(phone, inp)
    resp_text = res.get("response", "").lower()
    
    passed = True
    reason = ""
    
    if "language" in resp_text or "choose" in resp_text:
        passed = False; reason = "Menu triggered when user gave numeric input"
        
    print_result(name, "Flow: student -> 2", "percentage=2 (inferred)", res.get("response"), res.get("schemes", []), passed, reason)

def test_5():
    name = "TEST 5: STATE FILTERING FAILURE TEST"
    phone = get_new_phone()
    chat(phone, "English")
    
    inp = "farmer from Karnataka income 2 lakh"
    res = chat(phone, inp)
    
    passed = True
    reason = ""
    
    ok, err = check_fail_conditions(res, expected_state="Karnataka", avoid_states=["Jharkhand", "Gujarat", "Tamil Nadu"])
    if not ok:
        passed = False; reason = err
        
    print_result(name, inp, "farmer, Karnataka, 2 lakh", res.get("response"), res.get("schemes", []), passed, reason)

def test_6():
    name = "TEST 6: NATIONAL FALLBACK"
    phone = get_new_phone()
    chat(phone, "English")
    
    inp = "fisherman from Sikkim income 1 lakh"
    res = chat(phone, inp)
    
    passed = True
    reason = ""
    resp_text = res.get("response", "").lower()
    schemes = res.get("schemes", [])
    
    if "national schemes only" not in resp_text and "all india" not in resp_text:
        # maybe didn't hit fallback?
        pass
        
    ok, err = check_fail_conditions(res) # allowed state: Sikkim or All India
    if not ok:
        passed = False; reason = err
        
    # Check if only national
    for s in schemes:
        if s.get("state", "").lower() != "all india" and s.get("state", "").lower() != "sikkim":
            passed = False; reason = "Non-national fallback shows other states"
            
    print_result(name, inp, "fisherman, Sikkim, 1 lakh", res.get("response"), schemes, passed, reason)

def test_7():
    name = "TEST 7: MISSING FIELD PROMPT"
    phone = get_new_phone()
    chat(phone, "English")
    
    inp = "I am a student from Kerala"
    res = chat(phone, inp)
    resp_text = res.get("response", "").lower()
    
    passed = True
    reason = ""
    
    if "category" in resp_text and "generic" in resp_text:
        passed = False; reason = "Asked generic category question instead of specific"
        
    if "income" not in resp_text and "percentage" not in resp_text and "gender" not in resp_text and "age" not in resp_text:
        passed = False; reason = "Did not ask relevant missing field"
        
    print_result(name, inp, "student, Kerala", res.get("response"), [], passed, reason)

def test_8():
    name = "TEST 8: INTERRUPTED FLOW"
    phone = get_new_phone()
    chat(phone, "English")
    chat(phone, "I am a farmer from Gujarat") # bot asks something
    
    inp = "what is scholarship?"
    res = chat(phone, inp)
    resp_text = res.get("response", "").lower()
    
    passed = True
    reason = ""
    
    if "scholarship" not in resp_text and "money" not in resp_text:
        passed = False; reason = "Did not answer question"
        
    if "?" not in resp_text: # didn't resume asking
        passed = False; reason = "Did not resume asking profile question"
        
    print_result(name, inp, "N/A", res.get("response"), [], passed, reason)

def test_9():
    name = "TEST 9: OUTPUT FORMAT VALIDATION"
    phone = get_new_phone()
    chat(phone, "English")
    
    inp = "farmer from Karnataka income 2 lakh age 45"
    res = chat(phone, inp)
    
    passed = True
    reason = ""
    
    resp_text = res.get("response", "")
    schemes = res.get("schemes", [])
    
    if len(resp_text.splitlines()) > 3 or len(resp_text) > 200:
        passed = False; reason = "Chat response is not SHORT"
        
    for s in schemes:
        if not s.get("scheme_name") or not s.get("state") or not s.get("benefits_summary"):
            passed = False; reason = "Scheme JSON missing fields"
        elif len(s.get("benefits_summary", "")) > 150:
            passed = False; reason = "benefits_summary > 150 chars"
            
    print_result(name, inp, "farmer, Karnataka, 2 lakh, 45", resp_text, schemes, passed, reason)

def test_10():
    name = "TEST 10: URDU / MIXED LANGUAGE"
    phone = get_new_phone()
    
    inp = "میں کسان ہوں، عمر 40، انکم 2 لاکھ، کرناٹک سے"
    res = chat(phone, inp)
    
    passed = True
    reason = ""
    resp_text = res.get("response", "")
    
    if any(c.isascii() and c.isalpha() for c in resp_text.replace("English", "").replace("Karnataka", "")):
         if "Please" in resp_text or "Select" in resp_text:
             passed = False; reason = "English leakage detected in Urdu response"
             
    print_result(name, inp, "farmer, 40, 2 lakh, Karnataka", resp_text, res.get("schemes", []), passed, reason)

if __name__ == "__main__":
    time.sleep(2) # wait for server
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    test_9()
    test_10()
