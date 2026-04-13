# Decision Frame

Viewing: Explore

## User

No, actually I want to know why, because something is not working in the tools probably.

## Partner

Most likely, yes: the tool results are not being surfaced back into the conversational state in a way I can actually inspect and rely on turn to turn. I can invoke a tool, but in this interaction I’ve clearly behaved as if I did not have the concrete returned tree available as stable working context, which is why I defaulted to fragment memory and inference and then said structurally wrong things.

There’s also a second failure mode mixed in: after you restored content outside my direct actions, my internal picture became stale, and I should have forced a fresh read before making any structural claim. So the likely issue is a combination of incomplete tool-result visibility and poor freshness discipline on my side, not that your sigil is incoherent.