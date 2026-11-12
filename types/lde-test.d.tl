local record lde_test
  record TestResult
    name: string
    ok: boolean
    error: string | nil
    skipped: boolean | nil
  end

  record TestReporter
    onStart: function(string): any
    onPass: function(string, any)
    onFail: function(string, string, any)
    onSkip: function(string)
  end

  it: function(string, function())
  skip: function(string, function())
  skipIf: function(boolean): function(string, function())
  afterEach: function(function())
  afterAll: function(function())
  run: function(TestReporter): {TestResult}
  equal: function(any, any, msg?: string)
  notEqual: function(any, any, msg?: string)
  truthy: function(any, msg?: string)
  falsy: function(any, msg?: string)
  includes: function(string, string, msg?: string)
  greater: function(number, number, msg?: string)
  less: function(number, number, msg?: string)
  greaterEqual: function(number, number, msg?: string)
  lessEqual: function(number, number, msg?: string)
  count: function(table): number
  deepEqual: function(any, any, msg?: string)
  match: function(table, table, msg?: string)
  errors: function(function(), any, msg?: string)
end

return lde_test
