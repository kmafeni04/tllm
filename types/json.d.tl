local record json
  enum Type
    "raw"
    "null"
    "boolean"
    "number"
    "string"
    "array"
    "object"
  end
  record Null
  end

  type Scalar = string | number | boolean | Null
  record DecodeOptions
    json5: boolean | nil
  end

  record EncodeOptions
    indent: integer | nil
    escapeUnicode: boolean | nil
    escapeSlashes: boolean | nil
    newlineAtEnd: boolean | nil
    infNan: string | nil
    flags: integer | nil
  end

  record Value
    type: function(self: Value): Type
    is: function(self: Value, kind: Type): boolean
    size: function(self: Value): integer
    get: function(self: Value, key: string | integer, default?: any): any
    val: function(self: Value): Scalar | Value
    children: function(
      self: Value
    ): function(): integer | string | nil, Value | nil
    keys: function(self: Value): function(): string | nil
    values: function(self: Value): function(): boolean, Value | nil
    unpack: function(self: Value): any
  end

  record Document
    parse: function(source: string, opts?: DecodeOptions): Document
    open: function(path: string, opts?: DecodeOptions): Document
    root: function(self: Document): Value
    encode: function(self: Document, opts?: EncodeOptions): string
    stats: function(self: Document): integer, integer
    free: function(self: Document): nil
  end

  record edit
    open: function(path: string, opts?: DecodeOptions): edit.Document
    parse: function(text: string, opts?: DecodeOptions): edit.Document
    type Value = EditValue
    type Document = EditDocument
  end

  record EditValue
    type: function(self: EditValue): string
    is: function(self: EditValue, kind: Type): boolean
    size: function(self: EditValue): integer
    get: function(self: EditValue, key: string | integer): EditValue
    val: function(self: EditValue): Scalar | EditValue
    children: function(
      self: EditValue
    ): function(): integer | string | nil, EditValue | nil
    keys: function(self: EditValue): function(): string | nil
    values: function(self: EditValue): function(): boolean, EditValue | nil
    unpack: function(self: EditValue): any
    set: function(self: EditValue, key: string | integer, value: any): EditValue
    add: function(self: EditValue, key: string, value: any): EditValue
    append: function(self: EditValue, value: any): EditValue
    insert: function(self: EditValue, index: integer, value: any): EditValue
    remove: function(self: EditValue, key: string | integer): EditValue
  end

  record EditDocument
    path: string | nil
    parse: function(text: string, opts?: DecodeOptions): EditDocument
    import: function(doc: Document): EditDocument
    open: function(path: string, opts?: DecodeOptions): EditDocument
    docPtr: function(self: EditDocument): any
    root: function(self: EditDocument): EditValue
    toDocument: function(self: EditDocument): Document
    encode: function(self: EditDocument, opts?: EncodeOptions): string
    save: function(self: EditDocument, path?: string, opts?: EncodeOptions)
    free: function(self: EditDocument): nil
  end

  null: Null
  version: string
  encode: function(value: any, opts?: EncodeOptions): string
  decode: function(source: string, opts?: DecodeOptions): Value
  decodeFile: function(path: string, opts?: DecodeOptions): Value
end

return json
