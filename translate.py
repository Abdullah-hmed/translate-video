# Using requests to Google Translate web endpoint (not official API)
import requests

def quick_translate(text, target='en'):
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target}&dt=t&q={text}"
    return requests.get(url).json()[0][0][0]

def mtranslate(text, target='en', source='autodetect'):
    r = requests.get("https://api.mymemory.translated.net/get", params={
        "q": text,
        "langpair": f"{source}|{target}"
    })
    return r.json()["responseData"]["translatedText"]


if __name__ == "__main__":
    # print(quick_translate("Hello, how are you?"))
    print(mtranslate("Hello, how are you?", source="en", target="ur"))
