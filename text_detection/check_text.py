import tnkeeh as tn

# Initialize the Tnkeeh cleaner with desired parameters
cleaner = tn.Tnkeeh(
    remove_diacritics=True,
    remove_special_chars=True,
    remove_tatweel=True,
    remove_repeated_chars=True,
    remove_html_elements=True,
    remove_links=True,
    remove_twitter_meta=True,
    normalize=True,
    segment=False
)

# cleaned_text = cleaner.clean_string(text)