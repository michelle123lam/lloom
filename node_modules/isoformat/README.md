# isoformat

This library implements a concise formatter and parser for ISO 8601 date and date-times. It is intended for use as an interchange format, for example in CSV, that is more human-readable than the full ISO date-time string used by [*date*.toISOString](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/toISOString).

To use:

```js
import {format, parse} from "isoformat";
```

### format(*date*, *fallback*)

Given a Date, **format**(*date*) returns the shortest equivalent ISO 8601 UTC string. If *date* is not a Date instance, it is assumed to represent milliseconds since UNIX epoch. If *date* is not a valid date, returns the given *fallback* value, which defaults to undefined; if *fallback* is a function, it is invoked to produce a fallback value if needed, being passed the *date*.

```js
format(new Date(Date.UTC(2001, 0, 1))) // "2001-01-01"
format(new Date(Date.UTC(2020, 0, 1, 12, 23))) // "2020-01-01T12:23Z"
```

The following forms may be returned by format:

* YYYY-MM-DD
* YYYY-MM-DDTHH:MMZ
* YYYY-MM-DDTHH:MM:SSZ
* YYYY-MM-DDTHH:MM:SS.MMMZ

The year YYYY may also be represented as +YYYYYY or -YYYYYY. Note that while YYYY and YYYY-MM are valid ISO 8601 date strings, these forms are never returned by format; YYYY can be easily misinterpreted as a number, and YYYY-MM… well, I guess that would be okay, but it felt simpler to stop at YYYY-MM-DD to make it more obvious that it was a date.

### parse(*date*, *fallback*)

Given an ISO 8601 date or date-time string, **parse**(*string*) returns an equivalent Date instance. If *string* is not a valid ISO 8601 date or date-time string, returns the given *fallback* value, which defaults to undefined; if *fallback* is a function, it is invoked to produce a fallback value if needed, being passed the *string*.

```js
parse("2001-01-01") // new Date(Date.UTC(2001, 0, 1))
parse("2020-01-01T12:23Z") // new Date(Date.UTC(2020, 0, 1, 12, 23))
```

The following forms are accepted by parse:

* YYYY
* YYYY-MM
* YYYY-MM-DD
* YYYY-MM-DDTHH:MM
* YYYY-MM-DDTHH:MMZ
* YYYY-MM-DDTHH:MM:SS
* YYYY-MM-DDTHH:MM:SSZ
* YYYY-MM-DDTHH:MM:SS.MMM
* YYYY-MM-DDTHH:MM:SS.MMMZ

The year YYYY may also be represented as +YYYYYY or -YYYYYY. The time zone Z may be represented as a literal Z for UTC, or as +HH:MM, -HH:MM, +HHMM, or -HHMM. (The two-digit time zone offset +HH or -HH is not supported; although part of ISO 8601, this format is not recognized by Chrome or Node. And although ISO 8601 does not allow the time zone -00:00, it is allowed here because it is widely supported in implementations.)
