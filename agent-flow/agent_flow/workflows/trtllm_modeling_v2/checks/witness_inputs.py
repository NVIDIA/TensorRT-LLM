# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Witness workloads and word-counted text inputs; engine observations establish token lengths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..spec.task_schema import load_prompts_file
from .boot import BootSpec

_PASSAGE = """
The river town woke slowly on the first cold morning of autumn. Smoke rose from a few chimneys
and hung low over the water, and the ferry bell rang twice before anyone came down to the
landing. The ferryman, a patient man with grey hands, counted the crates of apples stacked by the
rail and decided that he would make the crossing whether or not the baker arrived. On the far
bank a line of poplars had already turned yellow, and their leaves drifted onto the current and
turned in slow circles as they went. A dog barked at the gulls. Somewhere up the hill a shutter
opened, and a woman called a name that nobody answered, so she called it again, louder, and then
closed the shutter with a bang.

The baker came running down the lane with flour still on his sleeves and a basket of loaves
balanced against his hip. He apologized to the ferryman, to the apples, and to the morning in
general, and he explained that his oven had refused to heat until he cleaned the flue, which he
had been meaning to do since the summer. The ferryman said nothing about it. He took the basket,
set it in the shade of the crates, and untied the rope from the post. The smell of warm bread
spread across the deck and made the two passengers who had been waiting in silence begin to talk
to each other about the price of wheat, the weather in the hills, and the new bridge that
everyone said would be built some day.

Halfway across, the current pulled harder than it had all week, and the ferryman leaned on the
long pole with his whole weight. He knew the river the way a farmer knows a field, by its moods
rather than its measurements. When the water ran brown and quick it meant rain in the mountains
two days earlier, and when it ran clear and lazy it meant the snow had not yet begun to melt.
Today it was brown. He steered upstream of the landing and let the river carry the boat down to
the posts, a trick his father had taught him and that he had never managed to explain to anyone
else. The boat bumped gently against the timbers, and the baker clapped, as if he had been
worried all along.

The lighthouse keeper wrote in his log every evening at the same hour, whether or not anything
had happened. Most entries were short. Wind from the west, moderate. Lamp lit at dusk, wick
trimmed twice. A fishing boat passed close to the reef at sunset and turned away in good time.
On stormy nights the entries grew longer, because he had to stay awake anyway and the writing
kept him company. He described the color of the waves, the sound the glass made when the gusts
struck it, and the way the beam seemed to bend in the rain. Years later his daughter found the
books in a trunk and read them all in one winter, and she said that she had never known her
father could write like that.

In the old orchard behind the church, the trees had been planted by people whose names were no
longer remembered. Some of them still produced apples of varieties that could not be found in
any shop, small and russet and sharp, or pale and soft with a taste like pears. A retired
teacher took it upon herself to save them. She learned to graft by reading a book from the
library and by ruining a dozen young branches before she succeeded. Each spring she cut thin
shoots from the oldest trees, bound them onto sturdy rootstock with tape and wax, and labeled
them with careful handwriting. By the time she was too old to climb a ladder, there were young
copies of every tree growing in gardens across the valley.

The clockmaker kept his shop open only in the afternoons, because he said that his eyes were
sharpest after lunch. The walls were covered with clocks that did not agree with each other, and
the room was filled with a soft argument of ticking. A customer brought in a pocket watch that
had belonged to her grandfather and had not run for forty years. The clockmaker opened the back,
looked at the movement through his loupe for a long time without speaking, and then said that
the mainspring was broken and the balance wheel was bent, but that the escapement was beautiful
and worth saving. He asked her to come back in a month. When she did, the watch was running, and
he refused to let her pay more than the cost of the spring.

The engineers who surveyed the gorge spent most of a summer hanging from ropes. They measured
the rock on both sides, tested how it cracked under hammers, and argued in the evenings about
whether a single arch would hold or whether the bridge should be suspended from cables. The
cables won in the end, because the rock on the eastern side was too soft to carry the thrust of
an arch. Towers were built on each rim, and a thin pilot line was carried across by a boy with a
kite on a windy afternoon, which became the most famous part of the story. After that came
heavier ropes, then wire, then cables as thick as a man, spun in place strand by strand over the
empty air.

Every year in late autumn the geese came down out of the north and settled on the marshes for a
few weeks before moving on. A group of volunteers counted them at dawn, standing in the reeds
with notebooks and cold fingers. Counting geese is harder than it sounds. They do not stay
still, they rise in waves when a hawk passes, and they land again in different places, so that
the same birds can be counted twice and others not at all. The volunteers developed methods of
their own, counting blocks of ten and estimating the rest, and comparing their numbers over hot
tea afterward. Over the decades the counts went up and down, and nobody could say for certain
why, but the records were kept, and that seemed to matter.

The library in the village had one room, one stove, and one librarian, who had worked there long
enough to know what most people would ask for before they reached the desk. In winter the room
was warm and smelled of paper and wood smoke, and the old men came to read newspapers that were
already a week out of date. A boy came every afternoon after school and sat in the corner by the
window with a book about volcanoes. He read it so many times that the spine cracked, and the
librarian quietly ordered a second copy without telling him. When he grew up he studied the
earth and its fires, and he sent the library a letter from a mountain on the other side of the
world.

The mountain railway climbed through a series of switchbacks so tight that the train had to
stop, reverse, and continue upward in the other direction several times before it reached the
pass. In winter the tracks ran through long wooden sheds built to keep the snow from burying
them. The engineer who drove the morning train had done it for most of his life, and he could
tell from the sound of the wheels whether the rails were icy or merely wet. Passengers who rode
with him often remembered the view more than anything else, the valley falling away beneath the
windows, the river reduced to a silver thread, and the villages small enough to cover with a
thumb. He no longer noticed the view, he said, but he noticed the people noticing it, and he
liked that.

The weaver kept her wool in baskets sorted by color, and most of the colors she made herself.
Onion skins gave a warm gold, walnut hulls a deep brown, and a certain lichen from the north side
of the rocks gave a soft purple that faded slowly in the sun. She worked at a wooden loom that
had belonged to her aunt, and she said it had opinions about which patterns it would allow.
Visitors who watched her work were often surprised by how much of the job was counting. Every
thread had to be in the right place before the first row was woven, and a single mistake at the
start would show up as a crooked line across the whole cloth. She did not mind the counting. She
said it was a kind of quiet.

At low tide the salt marsh became a different country. The channels drained to a trickle, the
mud shone in the sun, and thousands of small crabs came out of their holes to feed, retreating
sideways in a rustling wave whenever a shadow crossed them. The air smelled of salt and rot and
growing things, a smell that visitors disliked at first and later missed. Wading birds walked
carefully along the edges of the pools, stabbing at things too small for a person to see. A
family of otters lived in the bank of the largest creek, and on calm evenings they could be seen
sliding into the water one after another, as if they were playing a game whose rules only they
understood.

The beekeeper moved slowly around his hives, not because the bees were dangerous but because
sudden movements made them nervous, and nervous bees made poor honey. He used a little smoke
from a can of burning burlap to calm them before he lifted the lids. Inside, the frames were
heavy with comb, and the bees walked over his gloves without stinging. In spring a hive might
decide to swarm, and half the colony would leave with the old queen and settle in a dark cluster
on a branch nearby. When that happened he would put on his veil, climb a ladder with a box, and
shake the branch so that the whole swarm fell inside. Most of the time it worked. When it did
not, he laughed and tried again.

Before satellites, maps were made by people walking. A survey team would carry chains of a fixed
length across fields and through forests, measuring one span at a time and writing everything
down. They used instruments on tripods to measure angles from hilltop to hilltop, building a web
of triangles across the land. Errors crept in everywhere. A chain stretched in the heat, a stake
was moved by a cow, a number was copied wrong by a tired clerk. The best surveyors were not those
who made no mistakes but those who checked their work in ways that revealed the mistakes they
had made. When the triangles closed neatly at the end of a season, the whole team would
celebrate, often with more enthusiasm than the result deserved.

The small observatory stood on a hill outside the town, and on clear nights its dome turned
slowly as the telescope followed the stars. It was cold inside, colder than outside it seemed,
because the dome had to be open and the instruments could not be warmed without spoiling the
view. The astronomer who ran it was a cheerful woman who wore two coats and fingerless gloves and
drank tea from a flask. One winter she found a faint smudge near a familiar star that had not
been there the week before. She watched it for several nights, saw that it moved, and sent a
careful message to colleagues far away. It turned out to be a comet, and it was named after her,
which she found both pleasing and slightly embarrassing.

Along the flat coast, salt had been made the same way for centuries. Sea water was let into
shallow clay pans at high tide, and then the gates were closed and the sun and wind did the
rest. As the water evaporated, the brine grew thicker and moved from one pan to the next, until
finally crystals began to form on the surface like a thin skin of ice. Workers raked them
carefully toward the edges, drained them, and piled them in white mounds that shone in the
afternoon light. The best salt formed only on hot, dry days with a steady breeze from the land.
When the wind turned and blew in from the sea, the workers looked at the sky, shook their heads,
and went home early.

When the storm warnings went up, the harbor filled with activity. Fishermen doubled their
mooring lines, lashed down anything that could move, and took their nets ashore. Some of the
larger boats left the harbor altogether and ran for the deeper shelter of the bay, preferring
open water to the risk of being smashed against the stone quay. By nightfall the wind was howling
through the rigging, and waves broke over the harbor wall in great white sheets. A few men stayed
up all night in the harbor office, watching the boats through the rain and listening to the
radio. In the morning the sea was still rough, but the sky had cleared, and every boat was still
afloat, though one had lost its mast and another its dignity.

The printer set type by hand, picking each letter from a wooden case with compartments arranged
in a pattern that made no sense to anyone who had not learned it. Capital letters were kept in
the upper case and small letters in the lower case, which is where the names come from. He placed
the letters upside down and backward in a small metal holder, line by line, and then locked the
lines into a frame. A single page could take hours. When the frame was inked and pressed against
paper, the first proof was always full of small errors, and he would read it slowly with a
pencil, marking each one. He said that a printer learns humility from the first proof of every
job.

The winter market opened on the square a few weeks before the longest night of the year. Wooden
stalls were hung with lanterns and greenery, and the air smelled of roasted chestnuts, spiced
wine, and wet wool. Farmers sold cheese and smoked meat, carvers sold toys and spoons, and an old
woman sold knitted socks so thick that they could almost stand up on their own. In the evenings
a fiddler played near the fountain, and children danced in their heavy boots while their parents
pretended not to be cold. Every year someone complained that the market had become too crowded
and too expensive, and every year the same person came back, bought a bag of chestnuts, and stood
by the fiddler until the music stopped.

The potter wedged her clay on a heavy table, pressing and folding it until all the air was gone,
because a single bubble could crack a pot in the kiln. Then she threw a lump onto the center of
the wheel and wet her hands. Centering was the hardest part to teach. The clay had to spin
without the slightest wobble before it could be opened and raised, and beginners always fought
it, pushing too hard or not hard enough. She told her students to brace their elbows against
their knees and let the wheel do the work. Once the pots were dry, they were fired overnight in a
kiln that roared like a small dragon, and nobody knew how the glazes would turn out until it
cooled and the door was opened.

The village school had a single classroom where children of every age learned together, the
older ones helping the younger ones with their letters while the teacher worked through
arithmetic with the middle group. A stove in the corner burned wood that the families took turns
supplying, and on cold mornings the children arrived with logs under their arms. On the wall hung
a large map of the world, so old that some of the countries on it no longer existed and some of
the borders had been drawn by people who had never seen the places they divided. The teacher
used it anyway. She said it was a good lesson in itself, that maps are made by people and that
people change their minds.

Inside the glass studio, the furnace glowed day and night, because letting it cool and heating
it again took too long and cost too much. The glassblower dipped a long hollow pipe into the
molten glass and turned it slowly, gathering a bright orange blob on the end. Then she blew
gently into the pipe, rolled the glass on a steel table, and reheated it, over and over, shaping
it with wooden tools soaked in water that hissed and steamed at every touch. Her apprentice
dropped many pieces in his first year. She told him that every broken bowl was a lesson he would
not need to learn again, and he said that in that case he must be learning very quickly.

The fire tower stood on the highest ridge of the forest, a wooden cabin on stilts with windows on
every side. The ranger who lived there in summer spent her days scanning the horizon with
binoculars, looking for the thin grey line of smoke that meant a fire had started. Most days
there was nothing. She watched hawks circling below her, clouds building over the far mountains,
and the slow movement of shadows across the trees. When thunderstorms came, she counted the
seconds between each flash and the thunder and marked the strikes on a map, because lightning
that hit dry ground might smolder for days before it showed. She said the job taught her that
most of watching is waiting, and that waiting is a skill.

The lock keeper lived in a small stone house beside the canal, and his job was to raise and
lower boats between two levels of water. When a barge arrived from below, he opened the lower
gates, let it drift into the narrow chamber, and closed the gates behind it. Then he opened the
sluices in the upper gates, and water poured in, lifting the barge slowly until it floated level
with the canal above. The whole process took about a quarter of an hour, and most of the boatmen
used the time to talk. The lock keeper heard news from every town along the canal, often before
the newspapers did, and he passed it on to the next boat, sometimes improved a little in the
telling.

In early summer the shepherd took the flock up to the high pasture, a journey of two days along a
path that climbed through forest and then out onto open slopes covered with flowers. The dogs did
most of the work, running along the edges of the flock and bringing back the sheep that
wandered. At night they camped in a stone hut that had been used for this purpose for longer than
anyone could remember, and the shepherd made a fire and cooked beans in an iron pot. Up on the
pasture the air was thin and clean, and the only sounds were bells, wind, and the occasional cry
of a marmot. He stayed there until the first snow, and he said that the mountain made him a
quieter man.

The telegraph office in the desert town was a narrow room with a counter, a bench, and a
clattering machine that connected the town to the rest of the world. The operator could send and
receive messages faster than most people could speak, tapping out the code with a brass key and
listening to the replies with his eyes half closed. He knew every family business in town,
because sooner or later every important piece of news passed through his hands. Births, deaths,
debts, and engagements all came in as the same short and long sounds. He was trusted to keep
everything to himself, and he did, though people said that you could guess the news from the
look on his face as he walked across the square.

The boatbuilder steamed planks of oak in a long wooden box until they were soft enough to bend,
and then he and his son clamped them quickly against the curved ribs of the hull before they
cooled and stiffened. They worked fast and without speaking, because there was only a short time
before the wood would crack instead of bending. The hull grew slowly over the winter, plank by
plank, each one shaped by hand to fit against its neighbors. In spring they caulked the seams
with cotton and pitch, painted the hull, and launched it on a high tide with half the village
watching. The new boat floated a little lower at the stern than they had expected, and they spent
the summer arguing about why.

Rain came to the city in the late afternoon and stayed until night. Umbrellas opened along the
streets like a sudden field of dark flowers, and the gutters filled and ran with water that
carried leaves, paper, and the occasional lost glove. The tram windows fogged, and passengers
wiped small circles in the glass to see where they were. Under the awning of a closed shop, a
young man played a cello, and a few people stopped to listen even though they were getting wet.
He played slowly, as if the rain had set the tempo for him. When he finished, nobody clapped, but
several coins landed in his open case, and one woman left half of her sandwich, which he ate
gratefully between pieces.

The bookbinder folded large printed sheets into smaller sections, creased each fold with a smooth
piece of bone, and stacked the sections in order. Then she sewed them together along the spine
with linen thread, passing the needle through holes she had punched by hand. The sewing had to be
tight enough to hold the book together for a century and loose enough to let it open flat. She
glued the spine, rounded it with a hammer, and fitted it into covers of board and leather. The
last step was the lettering on the spine, pressed in with heated tools and thin gold leaf. She
said that a good binding was one nobody noticed, because the reader was too busy reading to think
about it.

The two mapmakers argued for most of a decade about a river that seemed to have moved. One of
them had surveyed it carefully and drawn it flowing west of the old mill. The other had surveyed
it just as carefully a few years later and found it flowing to the east. Each accused the other
of carelessness in letters that grew longer and less polite. In the end a third surveyor visited
the place and discovered that both of them had been right. A great flood in the years between
their visits had cut through a bend in the river and shifted its course by almost a mile. The two
men never apologized to each other, but they both quietly corrected their maps.

In the cool cellar under the farmhouse, the cheese sat on wooden shelves in long rows, each wheel
the size of a cart wheel and as heavy as a child. Every few days the cheesemaker turned them, one
by one, and washed their rinds with salt water to keep the wrong kinds of mold away. The work was
slow and repetitive, and it could not be hurried. A young cheese might be ready in a few months,
but the best ones were kept for two or three years, growing harder, darker, and stronger in
flavor. She tasted them with a long thin tool that drew out a small core from the center, and she
could tell from a single bite how much longer each wheel needed to wait.

The garden followed the year like a slow clock. In late winter the gardener turned the compost,
ordered seeds, and made plans that would mostly be abandoned by summer. In spring he planted peas
and potatoes, covered the young lettuce against frost, and fought a long war with slugs. Summer
brought more beans and zucchini than any household could eat, and he left baskets of them on the
doorsteps of neighbors who had stopped answering the door. In autumn he pulled out the dead
plants, saved seeds from the best tomatoes, and spread manure over the beds. Then the frost came
and the garden slept, and he sat by the window with a seed catalog, already imagining next year as
perfect.

When the lake froze thick enough to walk on, small wooden huts appeared on the ice, dragged out on
sledges by men who fished through holes they cut with long saws. Inside each hut a little stove
kept the cold away, and the fishermen sat for hours watching their lines and talking about
nothing in particular. At night the ice made strange sounds, deep booms and long singing notes
that rolled across the lake as it cracked and settled. Newcomers found the noises frightening,
but the old fishermen said the ice was only talking to itself. Every spring, someone left a hut
on the ice too long, and it sank through when the thaw came, and everyone said they had seen it
coming.

The old water mill had stopped grinding grain long ago, and now it was a house with a waterwheel
that turned for decoration. The family who lived there had found the millstones buried in the
yard and set them up as a table in the garden. An old man from the village visited once and told
them that he had worked in the mill as a boy. He remembered the noise above all, the rumble of the
stones and the splash of the wheel, so loud that the miller and his workers had communicated by
signs. He showed them where the grain had been stored, where the flour had come down, and where
the miller had kept a bottle hidden behind a loose board, which, to the delight of everyone, was
still there.

The caravan route crossed the desert from well to well, and the wells were the reason the route
existed at all. The travelers knew each one by name and by the taste of its water, sweet or salty
or faintly bitter. They walked in the early morning and the evening and rested through the heat
of the day under cloth stretched between poles. At night the guides navigated by the stars, which
in that dry air were so bright and numerous that the familiar patterns were almost lost among
them. The camels complained about everything and endured everything. When the caravan finally saw
the green line of palms at the edge of the oasis, nobody cheered, but everyone walked a little
faster.

The night train left the capital a little before midnight and arrived at the coast in the early
morning. Passengers slept in narrow bunks in compartments of four, and the gentle rocking of the
carriage put most of them to sleep within minutes. The conductor walked the corridor every hour
with a small lamp, checking that the doors were closed and that nobody was ill. Through the
windows, small towns flashed past in the dark, each one a cluster of streetlights and a single lit
window where someone else was also awake. At dawn the train ran along the shore, and the sea
appeared suddenly, grey and enormous, and the passengers who were already awake pressed their
faces to the glass.

The radio repair shop smelled of warm dust and solder. Shelves along the walls were stacked with
old sets waiting for parts that might never arrive, and the counter was covered with tubes,
wires, and small boxes of screws sorted by a system known only to the owner. He could diagnose
most faults by listening. A hum meant a failing capacitor, a crackle meant a loose connection,
and silence meant something more interesting. Customers brought him radios that had belonged to
their parents, and he understood that they were bringing him something more than electronics.
When a set came back to life on his bench and the first music came out of its speaker, he always
paused for a moment before he switched it off.

Once a year, at the end of the fishing season, the village held a festival for the boats. Each
family decorated its boat with flags, flowers, and colored paper, and the whole fleet sailed
slowly out of the harbor and around the headland while the people on shore sang songs that were
older than anyone could trace. A priest blessed the boats from the end of the pier, and children
threw flowers onto the water. In the evening there were tables in the square with fish grilled
over open fires, bread, and wine, and the singing continued long after dark. The old fishermen
said that the festival was for the sea as much as for the boats, a way of saying thank you and
please at the same time.

Building a dry stone wall requires no mortar, only patience and an eye for the shape of things.
The waller begins by digging a shallow trench and laying the largest stones along the bottom,
then builds upward in two faces that lean slightly inward toward each other. The space between
the faces is packed with small stones, and long through stones are laid across the wall at
intervals to tie the two faces together. The rule that every waller learns first is one over two,
meaning that each stone should cover the gap between the two stones beneath it. A well built wall
can stand for centuries. A badly built one falls down in the first hard winter, usually at night,
usually onto a neighbor.

The volcano observatory sat on a ridge within sight of the summit crater, and its instruments
listened to the mountain day and night. Seismometers recorded the small earthquakes that came in
swarms when the magma moved, tiltmeters measured how the ground swelled and sank, and sensors
sampled the gases drifting from vents in the rock. The scientists who worked there took turns
climbing to the crater rim to collect samples, wearing heavy boots and masks against the fumes.
Most of the time the mountain only grumbled. Their job was to tell the difference between
grumbling and something worse, and to give the towns on its slopes enough warning to leave. It
was a job, one of them said, in which being right quietly was the best possible outcome.

My sister wrote to me from a city on the other side of the ocean, where she had gone to study and
then stayed to work. Her letters arrived irregularly, sometimes three in a week and then nothing
for two months. She described the streets, the food, the strange politeness of strangers, and the
way the light fell differently in the afternoons there. She said that she had learned to dream in
another language and that it had surprised her more than anything else. She asked about the
river, the ferry, and the orchard behind the church, and whether the old teacher was still
grafting apple trees. I wrote back that she was, and that one of the young trees in our own
garden had flowered for the first time.

In the evening the river town settled into its usual quiet. The ferryman made his last crossing as
the sun went down behind the poplars, carrying a farmer, a goat, and a crate of empty bottles.
Lights came on in the windows up the hill, one after another, and the smell of cooking drifted
down to the water. The baker swept his floor and banked the fire in his oven for the morning. On
the far bank, the ferryman tied his boat to the post, coiled the rope, and stood for a moment
watching the current carry the last yellow leaves downstream. Then he walked home, past the
church and the orchard, and the town closed around him like a book at the end of a chapter.
"""


#: The passage as paragraphs, each one a tuple of its words.
_PARAGRAPHS: tuple[tuple[str, ...], ...] = tuple(
    tuple(block.split()) for block in _PASSAGE.strip().split("\n\n")
)


#: Every word of the passage in order, the first paragraph first.
_WORDS: tuple[str, ...] = tuple(word for paragraph in _PARAGRAPHS for word in paragraph)


#: Where each paragraph starts in :data:`_WORDS`.
_STARTS: tuple[int, ...] = tuple(
    sum(len(paragraph) for paragraph in _PARAGRAPHS[:index]) for index in range(len(_PARAGRAPHS))
)


def natural_text(count: int) -> str:
    """``count`` words of prose; actual token length depends on the tokenizer.

    The words run on from the start of paragraph ``count`` modulo the number of
    paragraphs, so two declared lengths are two texts, and neither finds its
    prefix cached by the other's prefill. A ``count`` over the passage's
    length repeats it.
    """
    if count <= 0:
        return ""
    start = _STARTS[count % len(_PARAGRAPHS)]
    return " ".join(_WORDS[(start + i) % len(_WORDS)] for i in range(count))


# Word counts size the workload, not its tokenization. Replay verdicts check
# the target's observed token count against the declared floor.
_FILLER = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


#: Generated tokens a token-comparing witness's continuations must reach,
#: when the block does not say. Identity over a 2-token answer rests on the
#: kernels' certification, not on an end-to-end comparison.
DEFAULT_MIN_GENERATED_TOKENS = 32


@dataclass(frozen=True)
class WitnessWorkload:
    """What a token-comparing witness sends, resolved from its block."""

    prompts: tuple[str, ...]
    max_tokens: int
    min_generated_tokens: int
    source: str


def resolve_witness_workload(
    task_data: Mapping[str, Any],
    block: Mapping[str, Any],
    boot: BootSpec,
    *,
    default_max_tokens: int,
) -> WitnessWorkload:
    """The witness block's own workload, else the declared acceptance prompts, else the boot prompts."""
    if block.get("prompts"):
        prompts, source = tuple(str(p) for p in block["prompts"]), "the witness block's prompts"
    elif block.get("prompts_file"):
        prompts = load_prompts_file(block["prompts_file"])
        source = f"the witness block's prompts_file ({block['prompts_file']})"
    else:
        acceptance = (task_data.get("accuracy") or {}).get("acceptance")
        acceptance = acceptance if isinstance(acceptance, Mapping) else {}
        if acceptance.get("prompts"):
            prompts = tuple(str(p) for p in acceptance["prompts"])
            source = "accuracy.acceptance.prompts"
        elif acceptance.get("prompts_file"):
            prompts = load_prompts_file(acceptance["prompts_file"])
            source = "accuracy.acceptance.prompts_file"
        else:
            prompts, source = boot.prompts, boot.source
    return WitnessWorkload(
        prompts=prompts,
        max_tokens=int(block.get("max_tokens") or default_max_tokens),
        min_generated_tokens=int(block.get("min_generated_tokens") or DEFAULT_MIN_GENERATED_TOKENS),
        source=source,
    )


def filler_words(count: int) -> str:
    """``count`` short words for replay witnesses; token length is observed at runtime."""
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(count))


__all__ = [
    "DEFAULT_MIN_GENERATED_TOKENS",
    "WitnessWorkload",
    "filler_words",
    "natural_text",
    "resolve_witness_workload",
]
