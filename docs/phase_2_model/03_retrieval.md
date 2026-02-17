## The Flow

Here's exactly what happens when you ask "What are the three Deathly Hallows?":

**Step 1:** Embed your question. Your question gets converted into a 384-dimensional vector using the same MiniLM model we used on the chunks. Now your question is a list of 384 numbers, just like every chunk.
**Step 2:** Compare against all 12,921 chunks. We calculate cosine similarity between your question vector and every single chunk vector. This is just math — dot product divided by magnitudes. It gives a score from 0 to 1 for each chunk. Higher means more semantically similar to your question.
**Step 3:** Grab the top 5. Sort all 12,921 scores, take the 5 highest. For the Deathly Hallows question, the top chunk scored 0.4917 — it happened to be the passage where Xenophilius draws the symbol and names all three objects. The retrieval found the right passage out of nearly 13,000 chunks.
**Step 4:** Build a prompt. We take those 5 passages, format them into a prompt that says "here's context from Harry Potter, answer this question using ONLY this context," and send it to Claude via the API.
**Step 5:** Claude reads and answers. Claude doesn't use its own Harry Potter knowledge. It reads those 5 specific passages like a reading comprehension test and formulates an answer.

### What the Output Tells Us

The results are revealing. Look at the difference between questions:

**Deathly Hallows** — worked great. The retriever found the exact passage where they're described. Claude gave a complete, accurate answer with citations. This is RAG working as intended.
**Half-Blood Prince** — partially worked. It found passages discussing the Prince but not the actual reveal (Snape). The chunks that discuss people theorizing about who the Prince is scored higher than the chunk where Dumbledore or Harry actually confirms it's Snape. Claude correctly said the answer wasn't fully in the provided context rather than making it up.
**Harry's scar** — failed. All 5 retrieved chunks were about the scar hurting, not about how he originally got it. The retriever matched on the word "scar" but grabbed the wrong passages. The origin story probably uses different language than "how did Harry get his scar" so the vectors didn't align well.
**Defeating Voldemort** — failed. Similar problem. The retriever found chunks about spells and Voldemort but not the specific final battle passage. Expelliarmus in the graveyard scene or the final duel might be described with different vocabulary than the question used.

### What This Tells Us About RAG

The system is only as good as the retrieval step. When it finds the right passages, the answers are excellent. When retrieval misses, even the smartest LLM can't help because we told it to only use the provided context. This is the core challenge of RAG — improving retrieval quality. More chunks (top_k=10 instead of 5), better chunking strategies, or hybrid search (keyword + semantic) would all help with the misses.
Want to make it interactive like we did with the text generator so you can play with it?


## Output

$ python3 src/models/03_retrieval.py
============================================================
PHASE 3: RAG RETRIEVAL + GENERATION
============================================================

Loading vector store...
✓ Loaded 12921 chunks with 384-dim vectors

Loading embedding model...
Loading weights: 100%|██| 103/103 [00:00<00:00, 4298.38it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
✓ Embedding model loaded

============================================================
TESTING RAG PIPELINE
============================================================

Question: What spell did Harry use to defeat Voldemort?
------------------------------------------------------------
Retrieving relevant passages...

Top 5 passages found:

  [1] Similarity: 0.6650
      e words Harry remembered, as though from a former life, the
dueling club at Hogwarts he had attended briefly two years ago. . . . All he
had learned t...

  [2] Similarity: 0.5969
      e helped him toward bed. Harry
nodded without any conviction and slumped back on his pillows, aching all
over from having fallen to the floor so often...

  [3] Similarity: 0.5930
      t him, and it wasn’t even a spell I recognized. I’ve never made gold
flames appear before.”
“Often,” said Mr. Weasley, “when you’re in a pressured sit...

  [4] Similarity: 0.5916
      y one. As I say, it was very
popular. . . . You know how these spells come and go. . . .”
“But it sounds like it was invented while you were at school...

  [5] Similarity: 0.5876
      Harry and
run . . . Voldemort had advanced on Lily Potter, told her to move aside so
that he could kill Harry . . . how she had begged him to kill her...

Generating answer...

Answer:
I couldn't find that in the provided passages. The passages show Harry facing Voldemort and discussing various spells, but they don't describe Harry actually defeating Voldemort with any particular spell. The passages mention Harry knowing "Expelliarmus" from dueling club and producing unexpected "gold flames," but there's no description of him using a spell to defeat Voldemort in these excerpts.

============================================================

Question: How did Harry get his scar?
------------------------------------------------------------
Retrieving relevant passages...

Top 5 passages found:

  [1] Similarity: 0.6842
      those touching her precious books.
Harry felt shivery; his scar was still aching, he felt almost feverish. When
he sat down opposite Ron and Hermione ...

  [2] Similarity: 0.6802
      the early hours of Saturday morning. All the curtains
were closed. As far as Harry could see through the darkness, there wasn’t a
living creature in s...

  [3] Similarity: 0.6686
      his scar searing with pain. Ron looked
as though he had just been getting ready for bed; one arm was out of his
robes.
“Has someone been attacked agai...

  [4] Similarity: 0.6678
      his scar was building
and he fought it: It was like resisting the urge to be sick.
He turned his back on Ron and Hermione, pretending to examine the o...

  [5] Similarity: 0.6676
      , so that it became a distant storm that reached Harry
from across a vast, silent ocean.
“I want to do it properly,” were the first words of which Har...

Generating answer...

Answer:
I couldn't find that in the provided passages. The passages mention Harry's scar multiple times - describing how it aches, burns, causes him pain, and connects him to Voldemort - but none of the passages explain how Harry originally got his scar.

============================================================

Question: What are the three Deathly Hallows?
------------------------------------------------------------
Retrieving relevant passages...

Top 5 passages found:

  [1] Similarity: 0.4917
      e of parchment from between more books.
“The Elder Wand,” he said, and he drew a straight vertical line upon the
parchment. “The Resurrection Stone,” ...

  [2] Similarity: 0.4831
      t some rest. What do you know about the
Deathly Hallows?”
“The — the what?” asked the wandmaker, looking utterly bewildered.
“The Deathly Hallows.”

“...

  [3] Similarity: 0.4784
      ’s no mention of the words ‘Deathly Hallows’ in the story,”
said Hermione.
“Well, of course not,” said Xenophilius, maddeningly smug. “That is a
child...

  [4] Similarity: 0.4711
      That was the name on the grave with the mark on it, in Godric’s
Hollow,” said Hermione, still watching Xenophilius. “Ignotus Peverell.”
“Exactly!” sai...

  [5] Similarity: 0.4682
      t the wrong time, for the wrong reasons. If you
laid hands on them, I wanted you to possess them safely. You are the true
master of death, because the...

Generating answer...

Answer:
Based on the provided passages, the three Deathly Hallows are:

1. **The Elder Wand** - described as a straight vertical line in the symbol
2. **The Resurrection Stone** - represented by a circle on top of the line in the symbol
3. **The Cloak of Invisibility** - represented by a triangle enclosing both the line and circle

This is explicitly stated in Passage 1, where Xenophilius draws the symbol and explains: "The Elder Wand," he said, and he drew a straight vertical line upon the parchment. "The Resurrection Stone," he said, and he added a circle on top of the line. "The Cloak of Invisibility," he finished, enclosing both line and circle in a triangle, to make the symbol that so intrigued Hermione. "Together," he said, "the Deathly Hallows."

Passage 3 further explains that these are "three objects, or Hallows, which, if united, will make the possessor master of Death."

============================================================

Question: Who is the Half-Blood Prince?
------------------------------------------------------------
Retrieving relevant passages...

Top 5 passages found:

  [1] Similarity: 0.6733
      about ‘the Prince’ as if it’s his
title, I bet it’s just a stupid nickname, and it doesn’t seem as though he was
a very nice person to me!”
“I don’t s...

  [2] Similarity: 0.6252
      r heard of someone called the Half-Blood Prince?”
“The Half-Blood what?”
“Prince,” said Harry, watching him closely for signs of recognition.
“There a...

  [3] Similarity: 0.6090
      name was Eileen Prince. Prince, Harry.”
They looked at each other, and Harry realized what Hermione was trying
to say. He burst out laughing.
“No way....

  [4] Similarity: 0.5945
      ggle,
then that would make her a ‘half-blood Prince’!”
“Yeah, very ingenious, Hermione . . .”
“But it would! Maybe she was proud of being half a Princ...

  [5] Similarity: 0.5250
      ple.”
“But you are normal!” said Harry fiercely. “You’ve just got a — a
problem —”
Lupin burst out laughing. “Sometimes you remind me a lot of James. ...

Generating answer...

Answer:
Based on the provided passages, the identity of the Half-Blood Prince is not definitively revealed. However, the passages provide several clues and theories:

**What we know for certain:**
- The Half-Blood Prince is someone who used to go to Hogwarts and owned an old Potions book
- He wrote spells in the book, including spells he invented like Levicorpus
- Harry is convinced the Prince is male, stating "I can tell it's not a girl. I can just tell" and "It's the way he writes, I just know the Prince was a bloke"

**Hermione's theory:**
The passages present Hermione's theory about someone named "Eileen Prince." As she explains: "If, say, her father was a wizard whose surname was Prince, and her mother was a Muggle, then that would make her a 'half-blood Prince'!" She suggests this could explain the title, saying "Maybe she was proud of being half a Prince!"

**What it's NOT:**
Lupin clarifies that "There are no Wizarding princes," suggesting it's not an actual royal title.

**The mystery remains:**
Harry dismisses Hermione's theory about Eileen Prince, and the passages don't provide the actual identity of the Half-Blood Prince. The question of who this person really is remains unanswered in these particular excerpts.

============================================================

## Output

$ python3 src/models/03_retrieval.py
============================================================
HARRY POTTER RAG Q&A SYSTEM
============================================================

Loading vector store...
✓ Loaded 12921 chunks with 384-dim vectors

Loading embedding model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██| 103/103 [00:00<00:00, 4371.19it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
✓ Embedding model loaded

============================================================
INTERACTIVE MODE
============================================================

Ask any Harry Potter question!
Commands:
  'quit'     - exit
  'top X'    - change number of chunks retrieved (default 5)
  'sources'  - toggle showing retrieved passages
------------------------------------------------------------

📖 Ask: How old was Harry when he started at Hogwarts?

Question: How old was Harry when he started at Hogwarts?
------------------------------------------------------------
Retrieving relevant passages...

Top 5 passages found:

  [1] Similarity: 0.6626
      will no doubt be aware, Harry comes of age in a year’s time —”
“No,” said Aunt Petunia, speaking for the first time since Dumbledore’s
arrival.
“I’m s...

  [2] Similarity: 0.5870
      been enough of a shock for Harry to discover, on his eleventh
birthday, that he was a wizard; it had been even more disconcerting to find
out that eve...

  [3] Similarity: 0.5689
      n as he could.
Outside in the hall, he leaned against the wall, breathing deeply. It had
been a long time since he’d lost control and made something e...

  [4] Similarity: 0.5546
      illed his
parents.
And then, exactly a year ago, Hogwarts had written to Harry, and the
whole story had come out. Harry had taken up his place at wiza...

  [5] Similarity: 0.5403
      me, however badly treated, you have at
least, grudgingly, allowed him houseroom. This magic will cease to operate
the moment that Harry turns seventee...

Generating answer...

Answer:
Based on the provided passages, I can determine Harry's age when he started Hogwarts by piecing together the information given.

From Passage 4: "And then, exactly a year ago, Hogwarts had written to Harry, and the whole story had come out. Harry had taken up his place at wizard school... The Dursleys hadn't even remembered that today happened to be Harry's twelfth birthday."

From Passage 2: "it had been enough of a shock for Harry to discover, on his eleventh birthday, that he was a wizard... At the end of this summer, he would be starting his fourth year at Hogwarts"

These passages indicate that Harry discovered he was a wizard on his eleventh birthday, and that he started at Hogwarts when he was eleven years old. Passage 4 shows he's currently twelve and reflecting on starting wizard school "exactly a year ago," which confirms he began at age eleven.

Therefore, Harry was eleven years old when he started at Hogwarts.

============================================================

📖 Ask: top 10
✓ Now retrieving top 10 chunks

📖 Ask:
