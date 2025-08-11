import { serve } from "https://deno.land/std@0.208.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.3'
import { corsHeaders } from '../_shared/cors.ts'

console.log("Event Ingestion Edge Function started")

interface UserEvent {
  user_id: string
  event_type: string
  properties: Record<string, any>
  timestamp?: string
  session_id?: string
  page_url?: string
  user_agent?: string
  ip_address?: string
  referrer?: string
}

interface ValidationError {
  field: string
  message: string
}

// Event validation
function validateEvent(event: any): { isValid: boolean; errors: ValidationError[] } {
  const errors: ValidationError[] = []

  // Required fields
  if (!event.user_id || typeof event.user_id !== 'string') {
    errors.push({ field: 'user_id', message: 'User ID is required and must be a string' })
  }

  if (!event.event_type || typeof event.event_type !== 'string') {
    errors.push({ field: 'event_type', message: 'Event type is required and must be a string' })
  }

  // Properties validation
  if (event.properties && typeof event.properties !== 'object') {
    errors.push({ field: 'properties', message: 'Properties must be an object' })
  }

  // Timestamp validation
  if (event.timestamp) {
    const timestamp = new Date(event.timestamp)
    if (isNaN(timestamp.getTime())) {
      errors.push({ field: 'timestamp', message: 'Invalid timestamp format' })
    }
  }

  // Check for SQL injection patterns
  const sqlInjectionPattern = /('|--|;|[|]|<|>|%|script|union|select|insert|update|delete|drop|create|alter)/i
  if (sqlInjectionPattern.test(event.user_id)) {
    errors.push({ field: 'user_id', message: 'User ID contains potentially harmful content' })
  }

  return {
    isValid: errors.length === 0,
    errors
  }
}

// Event enrichment
function enrichEvent(event: UserEvent, request: Request): UserEvent {
  const enrichedEvent = { ...event }

  // Add timestamp if not provided
  if (!enrichedEvent.timestamp) {
    enrichedEvent.timestamp = new Date().toISOString()
  }

  // Extract IP from request (Cloudflare header)
  if (!enrichedEvent.ip_address) {
    enrichedEvent.ip_address = request.headers.get('CF-Connecting-IP') || 
                               request.headers.get('X-Forwarded-For') || 
                               'unknown'
  }

  // Extract User-Agent
  if (!enrichedEvent.user_agent) {
    enrichedEvent.user_agent = request.headers.get('User-Agent') || 'unknown'
  }

  return enrichedEvent
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Initialize Supabase client
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    const url = new URL(req.url)
    const path = url.pathname

    // Health check endpoint
    if (path === '/health' || path === '/event-ingestion/health') {
      return new Response(
        JSON.stringify({ 
          status: 'healthy', 
          service: 'event-ingestion',
          timestamp: new Date().toISOString() 
        }),
        { 
          status: 200, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Single event endpoint
    if (path === '/events' || path === '/event-ingestion/events') {
      if (req.method !== 'POST') {
        return new Response(
          JSON.stringify({ error: 'Method not allowed' }),
          { 
            status: 405, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      const body = await req.json()
      
      // Validate event
      const { isValid, errors } = validateEvent(body)
      if (!isValid) {
        return new Response(
          JSON.stringify({ 
            status: 'error',
            message: 'Validation failed',
            errors 
          }),
          { 
            status: 400, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      // Enrich event
      const enrichedEvent = enrichEvent(body, req)

      // Store in Supabase
      const { data, error } = await supabaseClient
        .from('user_events')
        .insert([{
          user_id: enrichedEvent.user_id,
          event_type: enrichedEvent.event_type,
          properties: enrichedEvent.properties || {},
          timestamp: enrichedEvent.timestamp,
          session_id: enrichedEvent.session_id,
          page_url: enrichedEvent.page_url,
          user_agent: enrichedEvent.user_agent,
          ip_address: enrichedEvent.ip_address,
          referrer: enrichedEvent.referrer,
          processed_at: new Date().toISOString()
        }])

      if (error) {
        console.error('Database error:', error)
        return new Response(
          JSON.stringify({ 
            status: 'error',
            message: 'Failed to store event',
            error: error.message
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      return new Response(
        JSON.stringify({ 
          status: 'success',
          message: 'Event processed successfully',
          event_id: data?.[0]?.id
        }),
        { 
          status: 200, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Batch events endpoint
    if (path === '/events/batch' || path === '/event-ingestion/events/batch') {
      if (req.method !== 'POST') {
        return new Response(
          JSON.stringify({ error: 'Method not allowed' }),
          { 
            status: 405, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      const body = await req.json()
      const events = Array.isArray(body.events) ? body.events : [body]

      const processedEvents = []
      const errors = []

      for (const [index, event] of events.entries()) {
        const { isValid, errors: validationErrors } = validateEvent(event)
        if (!isValid) {
          errors.push({ index, errors: validationErrors })
          continue
        }

        const enrichedEvent = enrichEvent(event, req)
        processedEvents.push({
          user_id: enrichedEvent.user_id,
          event_type: enrichedEvent.event_type,
          properties: enrichedEvent.properties || {},
          timestamp: enrichedEvent.timestamp,
          session_id: enrichedEvent.session_id,
          page_url: enrichedEvent.page_url,
          user_agent: enrichedEvent.user_agent,
          ip_address: enrichedEvent.ip_address,
          referrer: enrichedEvent.referrer,
          processed_at: new Date().toISOString()
        })
      }

      if (processedEvents.length > 0) {
        const { data, error } = await supabaseClient
          .from('user_events')
          .insert(processedEvents)

        if (error) {
          console.error('Database error:', error)
          return new Response(
            JSON.stringify({ 
              status: 'error',
              message: 'Failed to store events',
              error: error.message
            }),
            { 
              status: 500, 
              headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
            }
          )
        }
      }

      return new Response(
        JSON.stringify({ 
          status: 'success',
          message: `Processed ${processedEvents.length} events successfully`,
          processed: processedEvents.length,
          errors: errors.length,
          validation_errors: errors
        }),
        { 
          status: 200, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Default 404 response
    return new Response(
      JSON.stringify({ error: 'Endpoint not found' }),
      { 
        status: 404, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )

  } catch (error) {
    console.error('Function error:', error)
    return new Response(
      JSON.stringify({ 
        status: 'error',
        message: 'Internal server error',
        error: error.message
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})
