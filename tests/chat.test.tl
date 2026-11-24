local fs = require("fs")
local json = require("json")
local test = require("lde-test")
local support = require("tests.support")
local chat = require("tllm.chat")
local tokenizer = require("tllm.tokenizer")
local u32, text, array = support.u32, support.text, support.array
local header, pair = support.header, support.pair

-- The chat template shipped by the model file under
-- ~/.local/share/llm-models/Qwen3.6-35B-A3B/, copied from
-- tokenizer.chat_template of Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf, with the
-- prompts that template writes for the plain text conversations T-001 names.
local TEMPLATE = [[{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set num_sys = 0 %}
{%- set merged_system = '' %}
{%- if messages[0].role == 'system' or messages[0].role == 'developer' %}
    {%- set first = render_content(messages[0].content, false, true)|trim %}
    {%- if messages|length > 1 and (messages[1].role == 'system' or messages[1].role == 'developer') %}
        {%- set second = render_content(messages[1].content, false, true)|trim %}
        {%- set merged_system = first + '\n' + second %}
        {%- set num_sys = 2 %}
    {%- else %}
        {%- set merged_system = first %}
        {%- set num_sys = 1 %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if merged_system %}
        {{- '\n\n' + merged_system }}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if merged_system %}
        {{- '<|im_start|>system\n' + merged_system + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if loop.index0 >= num_sys and message.role != "system" and message.role != "developer" %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is mapping %}
                    {%- for args_name in tool_call.arguments %}
                        {%- set args_value = tool_call.arguments[args_name] %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
{#- Unsloth fixes - developer role, tool calling #}]]

local WANT_SYSTEM = [[<|im_start|>system
be brief<|im_end|>
<|im_start|>user
hi<|im_end|>
<|im_start|>assistant
<think>
]]

local WANT_TWO_SYSTEM = [[<|im_start|>system
be brief
and plain<|im_end|>
<|im_start|>user
hi<|im_end|>
<|im_start|>assistant
<think>
]]

local WANT_THINK_SPLIT = [[<|im_start|>user
hi<|im_end|>
<|im_start|>assistant
<think>
plan
</think>

answer<|im_end|>
<|im_start|>assistant
<think>
]]

local THINK_OPEN = [[<think>]]
local THINK_CLOSE = [[</think>]]

local BRANCH_MARKERS = {
  [[<|image_start|>]],
  [[<|image_end|>]],
  [[<tool_call>]],
  [[<tool_response>]],
}

local record template_data
  chat: string
  want_system: string
  want_two_system: string
  want_think_split: string
  think_open: string
  think_close: string
  branch_markers: {string}
end

local fixture: template_data = {
  chat = TEMPLATE,
  want_system = WANT_SYSTEM,
  want_two_system = WANT_TWO_SYSTEM,
  want_think_split = WANT_THINK_SPLIT,
  think_open = THINK_OPEN,
  think_close = THINK_CLOSE,
  branch_markers = BRANCH_MARKERS,
}

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function strings(items: {string}): string
  local encoded: {string} = {}
  for _, item in ipairs(items) do
    encoded[#encoded + 1] = text(item)
  end
  return array(8, encoded)
end

local function ints(values: {integer}): string
  local encoded: {string} = {}
  for _, value in ipairs(values) do
    encoded[#encoded + 1] = u32(value)
  end
  return array(5, encoded)
end

local function gguf_bytes(extra: {string}): string
  local entries: {string} = {
    pair("tokenizer.ggml.model", 8, text("gpt2")),
    pair("tokenizer.ggml.pre", 8, text("gpt2")),
    pair("tokenizer.ggml.tokens", 9, strings({"a", "b"})),
    pair("tokenizer.ggml.token_type", 9, ints({1, 1})),
    pair("tokenizer.ggml.merges", 9, strings({})),
  }
  for _, entry in ipairs(extra) do
    entries[#entries + 1] = entry
  end
  created = created + 1
  local file_path = ROOT .. "/c" .. tostring(created) .. ".gguf"
  fs.write(file_path, header(0, #entries) .. table.concat(entries))
  return file_path
end

local function loaded(extra: {string}): tokenizer.Tokenizer
  local file_path = gguf_bytes(extra)
  local built, failure = tokenizer.load(file_path)
  test.equal(failure, nil, file_path)
  return built as tokenizer.Tokenizer
end

local function template_entry(template: string): string
  return pair("tokenizer.chat_template", 8, text(template))
end

local function with_template(template: string): tokenizer.Tokenizer
  return loaded({template_entry(template)})
end

local ONE: {chat.Message} = {{role = "user", content = "hi"}}

local PROMPT = "{% for message in messages %}<|im_start|>"
  .. "{{ message.role }}\n{{ message.content }}<|im_end|>\n"
  .. "{% endfor %}{% if add_generation_prompt %}"
  .. "<|im_start|>assistant\n{% endif %}"

local PROMPT_TEXT = "<|im_start|>system\nbe brief<|im_end|>\n"
  .. "<|im_start|>user\nhi<|im_end|>\n"
  .. "<|im_start|>assistant\n"

local function wrote(template: string, messages: {chat.Message}): string
  local prompt, failure = chat.render(with_template(template), messages)
  test.equal(failure, nil, template)
  return prompt as string
end

local function refused(template: string, items?: {chat.Message}): string
  local prompt, failure = chat.render(with_template(template), items or ONE)
  test.equal(prompt, nil, template)
  local line = failure as string
  test.truthy(line:find("tokenizer.chat_template: ", 1, true) == 1, line)
  test.equal(line:find("\n", 1, true), nil, line)
  return line
end

test.it(
  "messages become the prompt the template writes",
  function()
    local built = loaded({template_entry(PROMPT)})
    test.equal(built.template, PROMPT)

    local prompt, failure = chat.render(
      built,
      {
        {role = "system", content = "be brief"},
        {role = "user", content = "hi"},
      }
    )
    test.equal(failure, nil)
    test.equal(prompt, PROMPT_TEXT)
  end
)

test.it(
  "branches, trim markers and the trim filter shape the output",
  function()
    local trimmed = with_template(
      "a {%- if add_generation_prompt -%} b {%- endif -%} c"
    )
    local written, failure = chat.render(trimmed, ONE)
    test.equal(failure, nil)
    test.equal(written, "abc")

    local branched = with_template(
      "{% for message in messages %}"
        .. "{% if message.role == 'user' %}U"
        .. "{% elif message.role == 'assistant' %}A"
        .. "{% else %}S{% endif %}"
        .. "{{ message.content|trim }}{{ message.missing }}{% endfor %}"
    )
    written, failure = chat.render(
      branched,
      {
        {role = "user", content = " hi "},
        {role = "tool", content = "x"},
      }
    )
    test.equal(failure, nil)
    test.equal(written, "UhiSx")

    local numbered = with_template("a\n{{- 5 -}}\nc")
    written, failure = chat.render(numbered, ONE)
    test.equal(failure, nil)
    test.equal(written, "a5c")
  end
)

test.it(
  "a template or messages that cannot be used are refused",
  function()
    local bare = loaded({})
    test.equal(bare.template, nil)
    local written, failure = chat.render(bare, {})
    test.equal(written, nil)
    test.includes(failure as string, "tokenizer.chat_template")

    local numeric = loaded({pair("tokenizer.chat_template", 5, u32(7))})
    written, failure = chat.render(numeric, ONE)
    test.equal(written, nil)
    test.includes(failure as string, "tokenizer.chat_template")

    local macro = with_template("{% macro m() %}{% endmacro %}")
    written, failure = chat.render(macro, ONE)
    test.equal(failure, nil)
    test.equal(written, "")

    local _, unclosed_for = chat.render(
      with_template("{% for message in messages %}"), ONE
    )
    test.includes(unclosed_for as string, "tokenizer.chat_template")

    local _, bare_if = chat.render(with_template("{% if %}"), ONE)
    test.includes(bare_if as string, "tokenizer.chat_template")

    local _, stray_endif = chat.render(with_template("{% endif %}"), ONE)
    test.includes(stray_endif as string, "tokenizer.chat_template")

    local _, refused_filter = chat.render(with_template("{{ 'x'|upper }}"), ONE)
    test.includes(refused_filter as string, "tokenizer.chat_template")
    test.includes(refused_filter as string, "upper")
    local filter_line = refused_filter as string
    test.equal(filter_line:find("tojson", 1, true), nil, filter_line)

    local _, absent = chat.render(with_template("{{ missing.field }}"), ONE)
    test.includes(absent as string, "tokenizer.chat_template")

    local _, io_reach = chat.render(with_template("{{ io.open('x') }}"), ONE)
    test.includes(io_reach as string, "tokenizer.chat_template")

    local promised = loaded({template_entry(PROMPT)})
    local _, empty = chat.render(promised, {})
    test.truthy(empty)
    test.includes(empty as string, "messages")

    local broken = {role = "user", content = 5} as chat.Message
    local _, mixed = chat.render(promised, {ONE[1], broken})
    test.includes(mixed as string, "messages")
    test.includes(mixed as string, "2")

    local holed = {ONE[1], nil, ONE[1]} as {chat.Message}
    local _, gap = chat.render(promised, holed)
    test.includes(gap as string, "messages")
    test.includes(gap as string, "2")
  end
)

test.it(
  "the file's own template renders a text conversation",
  function()
    local built = with_template(fixture.chat)

    local prompt, failure = chat.render(
      built,
      {
        {role = "system", content = "be brief"},
        {role = "user", content = "hi"},
      }
    )
    test.equal(failure, nil)
    test.equal(prompt, fixture.want_system)

    prompt, failure = chat.render(
      built,
      {
        {role = "system", content = "be brief"},
        {role = "system", content = "and plain"},
        {role = "user", content = "hi"},
      }
    )
    test.equal(failure, nil)
    test.equal(prompt, fixture.want_two_system)

    prompt, failure = chat.render(
      built,
      {
        {role = "user", content = "hi"},
        {
          role = "assistant",
          content = "plan" .. fixture.think_close .. "answer",
        },
      }
    )
    test.equal(failure, nil)
    test.equal(prompt, fixture.want_think_split)

    local held = prompt as string
    local markers = fixture.branch_markers
    for index = 1, #markers do
      local marker = markers[index]
      test.equal(held:find(marker, 1, true), nil, marker)
    end
  end
)

test.it(
  "a set in a taken if outlives it and a set in a loop does not",
  function()
    test.equal(
      wrote(
        "{% set x = 0 %}{% if add_generation_prompt %}{% set y = 7 %}"
          .. "{% endif %}{{ y }}{% for m in messages %}{% set x = 5 %}"
          .. "{{ x }}{% endfor %}{{ x }}",
        ONE
      ),
      "750"
    )
  end
)

test.it(
  "namespace members survive the loop",
  function()
    test.equal(
      wrote(
        "{% set ns = namespace(seen=0, n=messages|length - 1) %}"
          .. "{% for m in messages %}{% set ns.seen = 1 %}{% endfor %}"
          .. "{{ ns.seen }}{{ ns.n }}{{ ns.absent }}",
        {{role = "user", content = "a"}, {role = "user", content = "b"}}
      ),
      "11"
    )
  end
)

test.it(
  "a template that cannot be run is one line",
  function()
    test.includes(refused("{{ raise_exception('bad text') }}"), "bad text")
    refused("{{ raise_exception() }}")
    refused("{{ raise_exception(3) }}")
    refused("{% include 'other' %}")
    local refused_line = refused("{{ 'x'|upper }}")
    test.includes(refused_line, "upper")
    test.equal(refused_line:find("tojson", 1, true), nil, refused_line)
    refused("{{ missing.field }}")
    refused("{{ io.open('x') }}")
    refused("{{ 'a' + 5 }}")
    refused("{% for m in missing %}{{ m }}{% endfor %}")
    refused("{{ 'x' in missing }}")
    test.equal(wrote("{{ missing }}", ONE), "")
  end
)

test.it(
  "values, methods and operators follow Python",
  function()
    test.equal(wrote("{% if '' %}A{% else %}B{% endif %}", ONE), "B")
    test.equal(wrote("{% if 0 %}A{% else %}B{% endif %}", ONE), "B")
    test.equal(wrote("{% if [] %}A{% else %}B{% endif %}", ONE), "B")
    test.equal(wrote("{% if ['a'] %}A{% else %}B{% endif %}", ONE), "A")
    test.equal(wrote("{% if namespace() %}T{% else %}F{% endif %}", ONE), "F")
    test.equal(
      wrote("{% if namespace(a=1) %}T{% else %}F{% endif %}", ONE), "T"
    )
    test.equal(
      wrote(
        "{{ ' a '.strip() }}{{ 'a-b'.split('-')[1] }}"
          .. "{{ 'abc'.startswith('ab') }}{{ 'abc'.replace('b', 'x') }}",
        ONE
      ),
      "abtrueaxc"
    )
    refused("{{ 'abc'.upper() }}")
    test.equal(
      wrote("{% for x in namespace(a=1) %}{{ x }}{% endfor %}", ONE), "a"
    )
    test.equal(
      wrote("{{ 'abc'[0] }}{{ 'abc'[-1] }}{{ 'abc'[::-1] }}", ONE), "accba"
    )
    test.equal(wrote("{{ 2 ~ '. ' }}", ONE), "2. ")
    local two: {chat.Message} = {
      {role = "user", content = "a"},
      {role = "user", content = "b"},
    }
    test.equal(
      wrote("{% for x in messages[::-1] %}{{ x.content }}{% endfor %}", two),
      "ba"
    )
    test.equal(
      wrote(
        "{% if 'b' in ['a', 'b'] %}1{% endif %}{% if 'b' in 'abc' %}2{% endif %}"
          .. "{% if 'b' not in 'ac' %}3{% endif %}",
        ONE
      ),
      "123"
    )
    test.equal(wrote("{{ messages[0].reasoning_content }}", ONE), "")
    test.equal(
      wrote(
        "{{ 'p1Cp2Cp3'.split('C')[0] }}{{ 'p1Cp2Cp3'.split('C')[-1] }}", ONE
      ),
      "p1p3"
    )
  end
)

test.it(
  "loop members follow Jinja",
  function()
    local body = "{{ loop.index0 }}|{{ loop.index }}|{{ loop.first }}"
      .. "|{{ loop.last }}|{{ loop.previtem }}|{{ loop.nextitem }}"
    local want = "0|1|true|false||b1|2|false|true|a|"
    test.equal(
      wrote("{% for x in ['a', 'b'] %}" .. body .. "{% endfor %}", ONE), want
    )
    test.equal(
      wrote("{% for x in 'ab' %}" .. body .. "{% endfor %}", ONE), want
    )
  end
)

test.it(
  "filters and comments shape the output",
  function()
    test.equal(wrote("{{ 'abc'|length }}", ONE), "3")
    test.equal(wrote("a {#- x -#} b", ONE), "ab")
    test.equal(wrote("a {# x #} b", ONE), "a  b")
  end
)

test.it(
  "an undefined answers the tests",
  function()
    local NAMES = {
      "defined",
      "undefined",
      "none",
      "string",
      "iterable",
      "mapping",
      "sequence",
      "true",
      "false",
    }
    local function hold(name: string): string
      local parts: {string} = {}
      for index, held in ipairs(NAMES) do
        parts[#parts + 1] = "{{ " .. name .. " is " .. held .. " }}"
        if index < #NAMES then
          parts[#parts + 1] = "|"
        end
      end
      return table.concat(parts)
    end
    test.equal(
      wrote(hold("missing"), ONE),
      "false|true|false|false|false|false|false|false|false"
    )
    test.equal(
      wrote("{% set n = 'x' %}" .. hold("n"), ONE),
      "true|false|false|true|true|false|true|false|false"
    )
    test.equal(wrote("{{ '' is string }}", ONE), "true")
  end
)

test.it(
  "a macro fills defaults, named arguments and its own value",
  function()
    test.equal(
      wrote(
        "{% macro m(a, b, c='z') %}{{ a }}{{ b }}{{ c }}{% endmacro %}"
          .. "{{ m('1', '2') }}",
        ONE
      ),
      "12z"
    )
    test.equal(
      wrote(
        "{% macro m(a, b, c='z') %}{{ a }}{{ b }}{{ c }}{% endmacro %}"
          .. "{{ m('1', c='3', b='2') }}",
        ONE
      ),
      "123"
    )
    test.equal(
      wrote("{% macro m(a) %}{{ a }}{% endmacro %}{{ m(' x ')|trim }}", ONE),
      "x"
    )
  end
)

local WANT = table.concat(
  {
    "<|im_start|>system\n",
    "# Tools\n",
    "\n",
    "You have access to the following functions:\n",
    "\n",
    "<tools>\n",
    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": []}}}\n",
    "</tools>\n",
    "\n",
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
    "\n",
    "<tool_call>\n",
    "<function=example_function_name>\n",
    "<parameter=example_parameter_1>\n",
    "value_1\n",
    "</parameter>\n",
    "<parameter=example_parameter_2>\n",
    "This is the value for the second parameter\n",
    "that can span\n",
    "multiple lines\n",
    "</parameter>\n",
    "</function>\n",
    "</tool_call>\n",
    "\n",
    "<IMPORTANT>\n",
    "Reminder:\n",
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
    "- Required parameters MUST be specified\n",
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
    "</IMPORTANT><|im_end|>\n",
    "<|im_start|>user\n",
    "hi<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n",
  },
  ""
)

-- The one-turn `hi` prompt: WANT's own tail, the lone turn and the opener.
local ONE_TURN = table.concat(
  {
    "<|im_start|>user\n",
    "hi<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n",
  },
  ""
)

-- WANT's entry line, and the two ends of the block around it, so another
-- entry list is proved in place rather than retyped whole.
local BLOCK_FROM = WANT:find("<tools>\n", 1, true) as integer
local BLOCK_TO = WANT:find("\n</tools>", 1, true) as integer
local WEATHER = WANT:sub(BLOCK_FROM + 8, BLOCK_TO - 1)

local function tools_prompt(entries: {string}): string
  return WANT:sub(1, BLOCK_FROM + 7)
    .. table.concat(entries, "\n")
    .. WANT:sub(BLOCK_TO)
end

local function wrote_tools(template: string, tools: json.Value): string
  local prompt, failure = chat.render(with_template(template), ONE, tools)
  test.equal(failure, nil, template)
  return prompt as string
end

local function wrote_plain(template: string): string
  local prompt, failure = chat.render(with_template(template), ONE)
  test.equal(failure, nil, template)
  return prompt as string
end

test.it(
  "a tool list becomes the tools block the file's template writes",
  function()
    local plain = wrote_plain(fixture.chat)
    test.equal(plain, ONE_TURN)
    test.equal(plain:find("<tools>", 1, true), nil)

    local one = wrote_tools(fixture.chat, json.decode("[" .. WEATHER .. "]"))
    test.equal(one, WANT)

    local add = '{"type": "function", "function": {"name": "add", "parameters": {}}}'
    local two = wrote_tools(
      fixture.chat, json.decode("[" .. WEATHER .. ", " .. add .. "]")
    )
    test.equal(two, tools_prompt({WEATHER, add}))

    local odd = wrote_tools(fixture.chat, json.decode("[5]"))
    test.equal(odd, tools_prompt({"5"}))

    test.equal(WANT:sub(#WANT - #ONE_TURN + 1), ONE_TURN)

    -- The tools block's system turn is the only turn before that one.
    local turns = 0
    local walk = 1
    while true do
      local at = WANT:find("<|im_start|>", walk, true)
      if at is nil then
        break
      end
      turns = turns + 1
      walk = (at as integer) + 1
    end
    test.equal(turns, 3)
  end
)

test.it(
  "which tool lists write no tools block, and which write one",
  function()
    test.equal(wrote_plain(fixture.chat), ONE_TURN)

    local NO_BLOCK = {"[]", "5", "true", "null", '{"type": "function"}'}
    for index = 1, #NO_BLOCK do
      local tools_json = NO_BLOCK[index]
      test.equal(
        wrote_tools(fixture.chat, json.decode(tools_json)), ONE_TURN, tools_json
      )
    end

    local chars = wrote_tools(fixture.chat, json.decode('"ab"'))
    test.equal(chars, tools_prompt({'"a"', '"b"'}))

    local hollow = wrote_tools(fixture.chat, json.decode("[[]]"))
    test.equal(hollow, tools_prompt({"[]"}))
  end
)

test.it(
  "tojson writes every value as JSON",
  function()
    local add = '{"type": "function", "function": {"name": "add", '
      .. '"parameters": {}}}'
    test.equal(
      wrote_tools(fixture.chat, json.decode("[" .. add .. "]")),
      tools_prompt({add})
    )

    local head = '{"type": "function", "function": {"name": "add", '
      .. '"parameters": {}'
    local defaulted = head .. ', "default": 1.0}}'
    local written = head .. ', "default": 1}}'
    test.equal(
      wrote_tools(fixture.chat, json.decode("[" .. defaulted .. "]")),
      tools_prompt({written})
    )

    local mixed = [==[[true, false, null, ["a", 2], {"b": [], "a": 2.5}]]==]
    test.equal(
      wrote_tools(fixture.chat, json.decode("[" .. mixed .. "]")),
      tools_prompt({mixed})
    )
  end
)

-- A template whose only output is the filter, over a tool list whose single
-- entry is the value under test.
local function wrote_json(entry: string): string
  local prompt, failure = chat.render(
    with_template("{{ tools[0] | tojson }}"),
    ONE,
    json.decode("[" .. entry .. "]")
  )
  test.equal(failure, nil, entry)
  return prompt as string
end

test.it(
  "every JSON kind, every escape and every key is written",
  function()
    test.equal(wrote_json([[{"b": 1, "a": 2}]]), [[{"b": 1, "a": 2}]])
    test.equal(wrote_json([==[{"a\"b\\c": 1}]==]), [==[{"a\"b\\c": 1}]==])
    test.equal(wrote_json([[{"k": ["x", "y"]}]]), [[{"k": ["x", "y"]}]])
    test.equal(wrote_json("{}"), "{}")
    test.equal(wrote_json("[]"), "[]")
    test.equal(wrote_json([==["a\"b\\c"]==]), [==["a\"b\\c"]==])
    test.equal(wrote_json([==["\t\n\r\b\f"]==]), [==["\t\n\r\b\f"]==])
    test.equal(wrote_json([==["\u0001"]==]), [==["\u0001"]==])
    test.equal(wrote_json([==["\u000E"]==]), [==["\u000e"]==])
    test.equal(wrote_json([==["\u001A"]==]), [==["\u001a"]==])
    test.equal(
      wrote_json([==["caf\u00e9"]==]), '"caf' .. string.char(0xC3, 0xA9) .. '"'
    )
    test.equal(wrote_json("2"), "2")
    test.equal(wrote_json("2.5"), "2.5")
    test.equal(wrote_json("true"), "true")
    test.equal(wrote_json("false"), "false")
    test.equal(wrote_json("null"), "null")

    local prompt, failure = chat.render(
      with_template("{{ tools[0].absent | tojson }}"),
      ONE,
      json.decode([==[[{"b": 1, "a": 2}]]==])
    )
    test.equal(failure, nil)
    test.equal(prompt, "null")
  end
)

test.it(
  "tojson is run and upper is not",
  function()
    local held = with_template("{{ 'x'|tojson }}")
    local written, failure = chat.render(held, ONE)
    test.equal(failure, nil)
    test.equal(written, '"x"')

    local line = refused("{{ 'x'|upper }}")
    test.includes(line, "tokenizer.chat_template")
    test.includes(line, "upper")
    test.equal(line:find("tojson", 1, true), nil, line)
  end
)

-- ONE_TURN's generation opener, and that opener with the empty thinking block
-- already closed: the two tails the file's own template writes.
local OPEN_TAIL = "<|im_start|>assistant\n<think>\n"
local CLOSED_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
local THINK_HEAD = ONE_TURN:sub(1, #ONE_TURN - #OPEN_TAIL)

local function wrote_flagged(template: string, flag: boolean): string
  local prompt, failure = chat.render(with_template(template), ONE, nil, flag)
  test.equal(failure, nil, template)
  return prompt as string
end

local ONLY_DEFINED = "{% if enable_thinking is defined %}t{% endif %}"
  .. "{% if preserve_thinking is defined %}p{% endif %}"

test.it(
  "the value a render call is given decides the closer the file's template writes",
  function()
    test.equal(wrote_flagged(fixture.chat, false), THINK_HEAD .. CLOSED_TAIL)
    test.equal(wrote_flagged(fixture.chat, true), ONE_TURN)
    test.equal(wrote_plain(fixture.chat), ONE_TURN)

    test.equal(wrote_flagged(ONLY_DEFINED, false), "t")
    test.equal(wrote_flagged(ONLY_DEFINED, true), "t")
    test.equal(wrote_plain(ONLY_DEFINED), "")
  end
)

test.it(
  "the tool-choice input binds its own name and no other",
  function()
    local built = with_template(
      "{% if tool_choice is defined %}t{% endif %}"
        .. "{% if preserve_thinking is defined %}p{% endif %}"
    )

    local written, failure = chat.render(
      built, ONE, nil, nil, json.decode('"auto"')
    )
    test.equal(failure, nil)
    test.equal(written, "t")

    written, failure = chat.render(built, ONE)
    test.equal(failure, nil)
    test.equal(written, "")
  end
)

-- 48/T-001: the call's arguments as the wire text holds them, the three turns
-- a harness replays after it, and the text the file's own template writes.

local CALL = '{"city": "Paris", "days": 2, "extra": {"unit": "c"}}'
local WANT_CALL = table.concat(
  {
    "<|im_start|>user\n",
    "hi<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n",
    "\n",
    "</think>\n\n",
    "<tool_call>\n",
    "<function=get_weather>\n",
    "<parameter=city>\n",
    "Paris\n",
    "</parameter>\n",
    "<parameter=days>\n",
    "2\n",
    "</parameter>\n",
    "<parameter=extra>\n",
    "{\"unit\": \"c\"}\n",
    "</parameter>\n",
    "</function>\n",
    "</tool_call><|im_end|>\n",
    "<|im_start|>user\n",
    "<tool_response>\n",
    "sunny\n",
    "</tool_response><|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n",
  },
  ""
)

local CALL_TURNS: {chat.Message} = {
  {role = "user", content = "hi"},
  {
    role = "assistant",
    content = "",
    tool_calls = {{name = "get_weather", arguments = json.decode(CALL)}},
  },
  {role = "tool", content = "sunny", tool_call_id = "call_1"},
}

test.it(
  "a tool call and its result render the turn the template writes",
  function()
    local built = with_template(fixture.chat)
    local prompt, failure = chat.render(built, CALL_TURNS)
    test.equal(failure, nil)
    test.equal(prompt, WANT_CALL)

    local named, named_failure = chat.render(
      with_template(
        "{% for message in messages %}"
          .. "{% if message.tool_call_id is defined %}{{ message.tool_call_id }}{% endif %}"
          .. "{% if message.name is defined %}{{ message.name }}{% endif %}"
          .. "{% endfor %}"
      ),
      {
        CALL_TURNS[1],
        CALL_TURNS[2],
        {
          role = "tool",
          content = "sunny",
          tool_call_id = "call_1",
          name = "get_weather",
        },
      }
    )
    test.equal(named_failure, nil)
    test.equal(named, "call_1get_weather")
  end
)

-- 48/T-002 and 48/T-006: the one assistant message whose call holds two keys.
local LOOKUP: {chat.Message} = {
  {
    role = "assistant",
    content = "",
    tool_calls = {
      {name = "lookup", arguments = json.decode('{"unit":"c","days":2}')},
    },
  },
}

test.it(
  "each call writes its name and every argument name",
  function()
    test.equal(
      wrote(
        "{% for message in messages %}"
          .. "{% for entry in message.tool_calls %}{{ entry.name }}"
          .. "{% for key in entry.arguments %}{{ key }},{% endfor %}"
          .. "{% endfor %}{% endfor %}",
        LOOKUP
      ),
      "lookupunit,days,"
    )
  end
)

-- 48/T-006: the two templates that hand the string filter a value it has no
-- text form for, over an argument map and over a list read off that map.
local WALK_ARGS = "{% for message in messages %}{% for entry in message.tool_calls %}"
  .. "{{ entry.arguments | string }}{% endfor %}{% endfor %}"

local WALK_LIST = "{% for message in messages %}{% for entry in message.tool_calls %}"
  .. "{{ entry.arguments.k | string }}{% endfor %}{% endfor %}"

local LIST_CALLS: {chat.Message} = {
  {
    role = "assistant",
    content = "",
    tool_calls = {
      {name = "lookup", arguments = json.decode('{"k":["a"]}')},
    },
  },
}

test.it(
  "the string filter refuses a map and a list",
  function()
    local map_line = refused(WALK_ARGS, LOOKUP)
    test.includes(map_line, "string")
    test.equal(refused(WALK_LIST, LIST_CALLS), map_line)
  end
)

-- 48/T-007: every kind the reader holds reaches a text form.
local KIND_CALLS: {chat.Message} = {
  {
    role = "assistant",
    content = "",
    tool_calls = {
      {
        name = "x",
        arguments = json.decode(
          '{"n": 1.0, "b": true, "f": false, "z": null}'
        ),
      },
    },
  },
}

test.it(
  "the string filter writes every kind the reader holds",
  function()
    test.equal(
      wrote(
        "{% for message in messages %}{% for entry in message.tool_calls %}"
          .. "{{ entry.arguments.n | string }}"
          .. "{{ entry.arguments.b | string }}"
          .. "{{ entry.arguments.f | string }}"
          .. "{{ entry.arguments.z | string }}"
          .. "{{ entry.arguments.absent | string }}"
          .. "{% endfor %}{% endfor %}",
        KIND_CALLS
      ),
      "1truefalsenullnull"
    )
  end
)
