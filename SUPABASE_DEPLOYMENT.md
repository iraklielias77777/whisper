# 🚀 Supabase Edge Functions Deployment Guide

## 📋 Prerequisites

1. **Supabase CLI**: Install globally
   ```bash
   npm install supabase -g
   ```

2. **Supabase Account**: Already set up at https://supabase.com/dashboard

## 🔧 Setup Instructions

### Step 1: Initialize Supabase Project
```bash
# Login to Supabase
supabase login

# Link to your existing project
supabase link --project-ref khkvmuctoqsamvggovov
```

### Step 2: Run Database Migrations
```bash
# Apply the schema migration
supabase db push
```

### Step 3: Deploy Edge Functions
```bash
# Deploy all functions
supabase functions deploy

# Or deploy individual functions
supabase functions deploy event-ingestion
supabase functions deploy behavioral-analysis
```

### Step 4: Set Environment Variables
```bash
# Set secrets for Edge Functions (replace with your actual keys)
supabase secrets set OPENAI_API_KEY=your_openai_api_key_here

supabase secrets set SENDGRID_API_KEY=your_sendgrid_api_key_here

supabase secrets set TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
supabase secrets set TWILIO_AUTH_TOKEN=your_twilio_auth_token_here

supabase secrets set FIREBASE_WEB_PUSH_KEY=your_firebase_web_push_key_here
```

> **🔐 Security Note**: Replace the placeholder values with your actual API keys from the previous setup.

## 🎯 Edge Function URLs

After deployment, your functions will be available at:

- **Event Ingestion**: `https://khkvmuctoqsamvggovov.supabase.co/functions/v1/event-ingestion`
- **Behavioral Analysis**: `https://khkvmuctoqsamvggovov.supabase.co/functions/v1/behavioral-analysis`
- **Decision Engine**: `https://khkvmuctoqsamvggovov.supabase.co/functions/v1/decision-engine`
- **Content Generation**: `https://khkvmuctoqsamvggovov.supabase.co/functions/v1/content-generation`
- **Channel Orchestrator**: `https://khkvmuctoqsamvggovov.supabase.co/functions/v1/channel-orchestrator`
- **AI Orchestration**: `https://khkvmuctoqsamvggovov.supabase.co/functions/v1/ai-orchestration`

## ✅ Testing

### Test Event Ingestion
```bash
curl -X POST https://khkvmuctoqsamvggovov.supabase.co/functions/v1/event-ingestion/events \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtoa3ZtdWN0b3FzYW12Z2dvdm92Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ3MzM0NzYsImV4cCI6MjA3MDMwOTQ3Nn0.PX-70uF9y6vtcMvavNMEB9IB3S1_jZnzz-2y7PixQ48" \
  -d '{
    "user_id": "test_user_123",
    "event_type": "page_view",
    "properties": {
      "page": "/dashboard",
      "category": "navigation"
    }
  }'
```

### Test Behavioral Analysis
```bash
curl -X POST https://khkvmuctoqsamvggovov.supabase.co/functions/v1/behavioral-analysis/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtoa3ZtdWN0b3FzYW12Z2dvdm92Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ3MzM0NzYsImV4cCI6MjA3MDMwOTQ3Nn0.PX-70uF9y6vtcMvavNMEB9IB3S1_jZnzz-2y7PixQ48" \
  -d '{
    "user_id": "test_user_123"
  }'
```

### Health Checks
```bash
# Check all services
curl https://khkvmuctoqsamvggovov.supabase.co/functions/v1/event-ingestion/health
curl https://khkvmuctoqsamvggovov.supabase.co/functions/v1/behavioral-analysis/health
```

## 🔒 Authentication

Edge Functions use Supabase's built-in authentication:
- **Anon Key**: For public endpoints
- **Service Role Key**: For administrative operations
- **JWT Tokens**: For user-specific operations

## 💰 Cost Benefits

- **No Database Connection Issues**: Direct access to Supabase DB
- **Auto-scaling**: Pay only for actual usage
- **Free Tier**: 500,000 function invocations/month
- **Global Edge**: Fast response times worldwide

## 🚀 Next Steps

1. Deploy the first two functions (event-ingestion, behavioral-analysis)
2. Test thoroughly
3. Migrate remaining services one by one
4. Update your frontend/SDK to use new Edge Function URLs

## 📞 Support

If you encounter issues:
1. Check Supabase function logs: `supabase functions logs`
2. Verify environment variables: `supabase secrets list`
3. Test locally: `supabase functions serve`
