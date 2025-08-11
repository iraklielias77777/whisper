import { serve } from "https://deno.land/std@0.208.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.3'
import { corsHeaders } from '../_shared/cors.ts'

console.log("Behavioral Analysis Edge Function started")

interface BehavioralMetrics {
  user_id: string
  session_count: number
  page_views: number
  session_duration_avg: number
  bounce_rate: number
  pages_per_session: number
  feature_usage_depth: number
  new_feature_adoption: number
  feature_stickiness: number
  most_used_features: string[]
  churn_risk_score: number
  engagement_score: number
}

// Calculate behavioral metrics from user events
async function calculateMetrics(supabase: any, userId: string): Promise<BehavioralMetrics> {
  // Get user events from last 30 days
  const thirtyDaysAgo = new Date()
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30)

  const { data: events, error } = await supabase
    .from('user_events')
    .select('*')
    .eq('user_id', userId)
    .gte('timestamp', thirtyDaysAgo.toISOString())
    .order('timestamp', { ascending: true })

  if (error) {
    throw new Error(`Failed to fetch user events: ${error.message}`)
  }

  if (!events || events.length === 0) {
    return {
      user_id: userId,
      session_count: 0,
      page_views: 0,
      session_duration_avg: 0,
      bounce_rate: 0,
      pages_per_session: 0,
      feature_usage_depth: 0,
      new_feature_adoption: 0,
      feature_stickiness: 0,
      most_used_features: [],
      churn_risk_score: 0.8, // High churn risk for inactive users
      engagement_score: 0
    }
  }

  // Group events by session
  const sessions = new Map<string, any[]>()
  events.forEach((event: any) => {
    const sessionId = event.session_id || 'default'
    if (!sessions.has(sessionId)) {
      sessions.set(sessionId, [])
    }
    sessions.get(sessionId)!.push(event)
  })

  // Calculate metrics
  const sessionCount = sessions.size
  const pageViews = events.filter((e: any) => e.event_type === 'page_view').length
  
  // Calculate average session duration
  let totalDuration = 0
  let validSessions = 0
  
  sessions.forEach((sessionEvents) => {
    if (sessionEvents.length > 1) {
      const start = new Date(sessionEvents[0].timestamp).getTime()
      const end = new Date(sessionEvents[sessionEvents.length - 1].timestamp).getTime()
      totalDuration += (end - start) / 1000 // Convert to seconds
      validSessions++
    }
  })
  
  const sessionDurationAvg = validSessions > 0 ? totalDuration / validSessions : 0
  
  // Calculate bounce rate (sessions with only 1 event)
  const bouncedSessions = Array.from(sessions.values()).filter(s => s.length === 1).length
  const bounceRate = sessionCount > 0 ? bouncedSessions / sessionCount : 0
  
  // Pages per session
  const pagesPerSession = sessionCount > 0 ? pageViews / sessionCount : 0
  
  // Feature usage analysis
  const featureEvents = events.filter((e: any) => e.event_type.startsWith('feature_'))
  const featureUsage = new Map<string, number>()
  
  featureEvents.forEach((event: any) => {
    const feature = event.event_type
    featureUsage.set(feature, (featureUsage.get(feature) || 0) + 1)
  })
  
  const mostUsedFeatures = Array.from(featureUsage.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([feature]) => feature)
  
  const featureUsageDepth = featureUsage.size
  const uniqueFeatures = featureUsage.size
  
  // Calculate engagement score (0-1)
  const recencyScore = 1 - Math.min(1, (Date.now() - new Date(events[events.length - 1].timestamp).getTime()) / (7 * 24 * 60 * 60 * 1000))
  const frequencyScore = Math.min(1, events.length / 100)
  const diversityScore = Math.min(1, uniqueFeatures / 10)
  const engagementScore = (recencyScore + frequencyScore + diversityScore) / 3
  
  // Calculate churn risk (inverse of engagement)
  const churnRiskScore = Math.max(0, 1 - engagementScore)
  
  return {
    user_id: userId,
    session_count: sessionCount,
    page_views: pageViews,
    session_duration_avg: sessionDurationAvg,
    bounce_rate: bounceRate,
    pages_per_session: pagesPerSession,
    feature_usage_depth: featureUsageDepth,
    new_feature_adoption: 0.5, // Would need historical data to calculate properly
    feature_stickiness: uniqueFeatures > 0 ? 0.7 : 0,
    most_used_features: mostUsedFeatures,
    churn_risk_score: churnRiskScore,
    engagement_score: engagementScore
  }
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    const url = new URL(req.url)
    const path = url.pathname

    // Health check
    if (path === '/health' || path === '/behavioral-analysis/health') {
      return new Response(
        JSON.stringify({ 
          status: 'healthy', 
          service: 'behavioral-analysis',
          timestamp: new Date().toISOString() 
        }),
        { 
          status: 200, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Analyze single user
    if (path === '/analyze' || path === '/behavioral-analysis/analyze') {
      if (req.method !== 'POST') {
        return new Response(
          JSON.stringify({ error: 'Method not allowed' }),
          { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      const body = await req.json()
      const { user_id } = body

      if (!user_id) {
        return new Response(
          JSON.stringify({ error: 'user_id is required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      try {
        const metrics = await calculateMetrics(supabaseClient, user_id)
        
        // Store metrics in database
        const { data, error } = await supabaseClient
          .from('behavioral_metrics')
          .upsert([{
            user_id: metrics.user_id,
            session_count: metrics.session_count,
            page_views: metrics.page_views,
            session_duration_avg: metrics.session_duration_avg,
            bounce_rate: metrics.bounce_rate,
            pages_per_session: metrics.pages_per_session,
            feature_usage_depth: metrics.feature_usage_depth,
            new_feature_adoption: metrics.new_feature_adoption,
            feature_stickiness: metrics.feature_stickiness,
            most_used_features: metrics.most_used_features,
            churn_risk_score: metrics.churn_risk_score,
            engagement_score: metrics.engagement_score,
            calculated_at: new Date().toISOString()
          }], {
            onConflict: 'user_id'
          })

        if (error) {
          console.error('Database error:', error)
          return new Response(
            JSON.stringify({ 
              status: 'error',
              message: 'Failed to store metrics',
              error: error.message
            }),
            { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
          )
        }

        return new Response(
          JSON.stringify({ 
            status: 'success',
            user_id,
            metrics,
            timestamp: new Date().toISOString()
          }),
          { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )

      } catch (error) {
        console.error('Analysis error:', error)
        return new Response(
          JSON.stringify({ 
            status: 'error',
            message: 'Analysis failed',
            error: error.message
          }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }
    }

    // Batch analysis
    if (path === '/analyze/batch' || path === '/behavioral-analysis/analyze/batch') {
      if (req.method !== 'POST') {
        return new Response(
          JSON.stringify({ error: 'Method not allowed' }),
          { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      const body = await req.json()
      const { user_ids } = body

      if (!Array.isArray(user_ids) || user_ids.length === 0) {
        return new Response(
          JSON.stringify({ error: 'user_ids array is required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      const results = []
      const errors = []

      for (const userId of user_ids) {
        try {
          const metrics = await calculateMetrics(supabaseClient, userId)
          
          // Store metrics
          await supabaseClient
            .from('behavioral_metrics')
            .upsert([{
              user_id: metrics.user_id,
              session_count: metrics.session_count,
              page_views: metrics.page_views,
              session_duration_avg: metrics.session_duration_avg,
              bounce_rate: metrics.bounce_rate,
              pages_per_session: metrics.pages_per_session,
              feature_usage_depth: metrics.feature_usage_depth,
              new_feature_adoption: metrics.new_feature_adoption,
              feature_stickiness: metrics.feature_stickiness,
              most_used_features: metrics.most_used_features,
              churn_risk_score: metrics.churn_risk_score,
              engagement_score: metrics.engagement_score,
              calculated_at: new Date().toISOString()
            }], {
              onConflict: 'user_id'
            })

          results.push({ user_id: userId, metrics })
        } catch (error) {
          errors.push({ user_id: userId, error: error.message })
        }
      }

      return new Response(
        JSON.stringify({ 
          status: 'success',
          processed: results.length,
          errors: errors.length,
          results,
          error_details: errors,
          timestamp: new Date().toISOString()
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    return new Response(
      JSON.stringify({ error: 'Endpoint not found' }),
      { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Function error:', error)
    return new Response(
      JSON.stringify({ 
        status: 'error',
        message: 'Internal server error',
        error: error.message
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
