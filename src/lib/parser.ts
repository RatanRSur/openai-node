import {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
} from '../resources/chat/completions';
import {
  ChatCompletionStreamingToolRunnerParams,
  ChatCompletionStreamParams,
  ChatCompletionToolRunnerParams,
  ParsedChatCompletion,
  ParsedChoice,
  ParsedFunctionToolCall,
} from '../resources/beta/chat/completions';
import { ResponseFormatJSONSchema } from '../resources/shared';
import { ContentFilterFinishReasonError, LengthFinishReasonError, OpenAIError } from '../error';
import { type ResponseFormatTextJSONSchemaConfig } from '../resources/responses/responses';
import { debug } from '../core';

type AnyChatCompletionCreateParams =
  | ChatCompletionCreateParams
  | ChatCompletionToolRunnerParams<any>
  | ChatCompletionStreamingToolRunnerParams<any>
  | ChatCompletionStreamParams;

export type ExtractParsedContentFromParams<Params extends AnyChatCompletionCreateParams> =
  Params['response_format'] extends AutoParseableResponseFormat<infer P> ? P : null;

export type AutoParseableResponseFormat<ParsedT> = ResponseFormatJSONSchema & {
  __output: ParsedT; // type-level only

  $brand: 'auto-parseable-response-format';
  $parseRaw(content: string): ParsedT;
};

export function makeParseableResponseFormat<ParsedT>(
  response_format: ResponseFormatJSONSchema,
  parser: (content: string) => ParsedT,
): AutoParseableResponseFormat<ParsedT> {
  const obj = { ...response_format };

  Object.defineProperties(obj, {
    $brand: {
      value: 'auto-parseable-response-format',
      enumerable: false,
    },
    $parseRaw: {
      value: parser,
      enumerable: false,
    },
  });

  return obj as AutoParseableResponseFormat<ParsedT>;
}

export type AutoParseableTextFormat<ParsedT> = ResponseFormatTextJSONSchemaConfig & {
  __output: ParsedT; // type-level only

  $brand: 'auto-parseable-response-format';
  $parseRaw(content: string): ParsedT;
};

export function makeParseableTextFormat<ParsedT>(
  response_format: ResponseFormatTextJSONSchemaConfig,
  parser: (content: string) => ParsedT,
): AutoParseableTextFormat<ParsedT> {
  const obj = { ...response_format };

  Object.defineProperties(obj, {
    $brand: {
      value: 'auto-parseable-response-format',
      enumerable: false,
    },
    $parseRaw: {
      value: parser,
      enumerable: false,
    },
  });

  return obj as AutoParseableTextFormat<ParsedT>;
}

export function isAutoParsableResponseFormat<ParsedT>(
  response_format: any,
): response_format is AutoParseableResponseFormat<ParsedT> {
  return response_format?.['$brand'] === 'auto-parseable-response-format';
}

type ToolOptions = {
  name: string;
  arguments: any;
  function?: ((args: any) => any) | undefined;
};

export type AutoParseableTool<
  OptionsT extends ToolOptions,
  HasFunction = OptionsT['function'] extends Function ? true : false,
> = ChatCompletionTool & {
  __arguments: OptionsT['arguments']; // type-level only
  __name: OptionsT['name']; // type-level only
  __hasFunction: HasFunction; // type-level only

  $brand: 'auto-parseable-tool';
  $callback: ((args: OptionsT['arguments']) => any) | undefined;
  $parseRaw(args: string): OptionsT['arguments'];
};

export function makeParseableTool<OptionsT extends ToolOptions>(
  tool: ChatCompletionTool,
  {
    parser,
    callback,
  }: {
    parser: (content: string) => OptionsT['arguments'];
    callback: ((args: any) => any) | undefined;
  },
): AutoParseableTool<OptionsT['arguments']> {
  const obj = { ...tool };

  Object.defineProperties(obj, {
    $brand: {
      value: 'auto-parseable-tool',
      enumerable: false,
    },
    $parseRaw: {
      value: parser,
      enumerable: false,
    },
    $callback: {
      value: callback,
      enumerable: false,
    },
  });

  return obj as AutoParseableTool<OptionsT['arguments']>;
}

export function isAutoParsableTool(tool: any): tool is AutoParseableTool<any> {
  return tool?.['$brand'] === 'auto-parseable-tool';
}

export function maybeParseChatCompletion<
  Params extends ChatCompletionCreateParams | null,
  ParsedT = Params extends null ? null : ExtractParsedContentFromParams<NonNullable<Params>>,
>(completion: ChatCompletion, params: Params): ParsedChatCompletion<ParsedT> {
  if (!params || !hasAutoParseableInput(params)) {
    return {
      ...completion,
      choices: completion.choices.map((choice) => ({
        ...choice,
        message: {
          ...choice.message,
          parsed: null,
          ...(choice.message.tool_calls ?
            {
              tool_calls: choice.message.tool_calls,
            }
          : undefined),
        },
      })),
    };
  }

  return parseChatCompletion(completion, params);
}

export function parseChatCompletion<
  Params extends ChatCompletionCreateParams,
  ParsedT = ExtractParsedContentFromParams<Params>,
>(completion: ChatCompletion, params: Params): ParsedChatCompletion<ParsedT> {
  debug('parseChatCompletion', 'Starting to parse chat completion');
  debug('parseChatCompletion:input', { completion, params });
  debug('parseChatCompletion:choices', `Processing ${completion.choices.length} choices`);

  const choices: Array<ParsedChoice<ParsedT>> = completion.choices.map((choice, index): ParsedChoice<ParsedT> => {
    debug('parseChatCompletion:choice', { index, choice });
    debug('parseChatCompletion:finish_reason', choice.finish_reason);

    if (choice.finish_reason === 'length') {
      debug('parseChatCompletion:error', 'Length finish reason detected, throwing LengthFinishReasonError');
      throw new LengthFinishReasonError();
    }

    if (choice.finish_reason === 'content_filter') {
      debug('parseChatCompletion:error', 'Content filter finish reason detected, throwing ContentFilterFinishReasonError');
      throw new ContentFilterFinishReasonError();
    }

    const hasToolCalls = Boolean(choice.message.tool_calls);
    debug('parseChatCompletion:tool_calls', { hasToolCalls });
    if (hasToolCalls) {
      debug('parseChatCompletion:tool_calls:data', choice.message.tool_calls);
    }

    const hasContent = Boolean(choice.message.content);
    const hasRefusal = Boolean(choice.message.refusal);
    debug('parseChatCompletion:content', { hasContent, hasRefusal });

    if (hasContent) {
      debug('parseChatCompletion:content:data', choice.message.content);
    }

    let parsedToolCalls: ParsedFunctionToolCall[] | undefined;
    if (hasToolCalls) {
      debug('parseChatCompletion:parsing_tool_calls', 'Starting to parse tool calls');
      parsedToolCalls = choice.message.tool_calls?.map((toolCall, toolIndex) => {
        debug('parseChatCompletion:tool_call', { toolIndex, toolCall });
        const parsed = parseToolCall(params, toolCall);
        debug('parseChatCompletion:tool_call:parsed', { toolIndex, parsed });
        return parsed;
      });
    }

    let parsedContent = null;
    if (hasContent && !hasRefusal) {
      debug('parseChatCompletion:parsing_content', 'Attempting to parse response format');
      parsedContent = parseResponseFormat(params, choice.message.content as string) as ParsedT | null;
      debug('parseChatCompletion:content:parsed', parsedContent);
    }

    const result = {
      ...choice,
      message: {
        ...choice.message,
        ...(choice.message.tool_calls ? {
          tool_calls: parsedToolCalls as ParsedFunctionToolCall[],
        } : {}),
        parsed: parsedContent,
      },
    };

    debug('parseChatCompletion:choice:result', { index, result });
    return result;
  });

  const result = { ...completion, choices };
  debug('parseChatCompletion:result', result);
  return result;
}

function parseResponseFormat<
  Params extends ChatCompletionCreateParams,
  ParsedT = ExtractParsedContentFromParams<Params>,
>(params: Params, content: string): ParsedT | null {
  if (params.response_format?.type !== 'json_schema') {
    return null;
  }

  if (params.response_format?.type === 'json_schema') {
    if ('$parseRaw' in params.response_format) {
      const response_format = params.response_format as AutoParseableResponseFormat<ParsedT>;

      return response_format.$parseRaw(content);
    }

    return JSON.parse(content);
  }

  return null;
}

function parseToolCall<Params extends ChatCompletionCreateParams>(
  params: Params,
  toolCall: ChatCompletionMessageToolCall,
): ParsedFunctionToolCall {
  const inputTool = params.tools?.find((inputTool) => inputTool.function?.name === toolCall.function.name);
  return {
    ...toolCall,
    function: {
      ...toolCall.function,
      parsed_arguments:
        isAutoParsableTool(inputTool) ? inputTool.$parseRaw(toolCall.function.arguments)
        : inputTool?.function.strict ? JSON.parse(toolCall.function.arguments)
        : null,
    },
  };
}

export function shouldParseToolCall(
  params: ChatCompletionCreateParams | null | undefined,
  toolCall: ChatCompletionMessageToolCall,
): boolean {
  if (!params) {
    return false;
  }

  const inputTool = params.tools?.find((inputTool) => inputTool.function?.name === toolCall.function.name);
  return isAutoParsableTool(inputTool) || inputTool?.function.strict || false;
}

export function hasAutoParseableInput(params: AnyChatCompletionCreateParams): boolean {
  if (isAutoParsableResponseFormat(params.response_format)) {
    return true;
  }

  return (
    params.tools?.some(
      (t) => isAutoParsableTool(t) || (t.type === 'function' && t.function.strict === true),
    ) ?? false
  );
}

export function validateInputTools(tools: ChatCompletionTool[] | undefined) {
  for (const tool of tools ?? []) {
    if (tool.type !== 'function') {
      throw new OpenAIError(
        `Currently only \`function\` tool types support auto-parsing; Received \`${tool.type}\``,
      );
    }

    if (tool.function.strict !== true) {
      throw new OpenAIError(
        `The \`${tool.function.name}\` tool is not marked with \`strict: true\`. Only strict function tools can be auto-parsed`,
      );
    }
  }
}
