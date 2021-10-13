import sanitizer

two_letter_words = set()
three_letter_words = set()

war_and_peace = open('./sources/WarAndPeace.txt').read()
sanitized_text = sanitizer.sanitize(war_and_peace)

words = sanitized_text.split(' ')

for word in words:
  if len(word) == 2:
    two_letter_words.add(word.lower())
  elif len(word) == 3:
    three_letter_words.add(word.lower())
