import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Save } from 'lucide-react'

export default function CampaignCreate() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    name: '',
    budget: '',
    dailyBudget: '',
    biddingStrategy: 'cpc',
    bidAmount: '',
    adFormat: 'banner',
    startDate: '',
    endDate: '',
    targeting: {
      countries: [] as string[],
      devices: [] as string[],
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Creating campaign:', formData)
    // API call would go here
    navigate('/campaigns')
  }

  return (
    <div className="max-w-3xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/campaigns')}
          className="flex items-center text-sm text-gray-500 hover:text-gray-700"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to Campaigns
        </button>
        <h1 className="mt-2 text-2xl font-bold text-gray-900">Create New Campaign</h1>
        <p className="mt-1 text-sm text-gray-500">Set up your advertising campaign parameters</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6 bg-white shadow rounded-lg p-6">
        {/* Basic Info */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Basic Information</h3>
          <div className="grid grid-cols-1 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Campaign Name</label>
              <input
                type="text"
                required
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="e.g., Summer Sale 2024"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
              />
            </div>
          </div>
        </div>

        {/* Budget */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Budget</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Total Budget ($)</label>
              <input
                type="number"
                required
                min="1"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="1000"
                value={formData.budget}
                onChange={(e) => setFormData({...formData, budget: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Daily Budget ($)</label>
              <input
                type="number"
                min="1"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="100"
                value={formData.dailyBudget}
                onChange={(e) => setFormData({...formData, dailyBudget: e.target.value})}
              />
            </div>
          </div>
        </div>

        {/* Bidding */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Bidding Strategy</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Strategy</label>
              <select
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                value={formData.biddingStrategy}
                onChange={(e) => setFormData({...formData, biddingStrategy: e.target.value})}
              >
                <option value="cpc">CPC (Cost per Click)</option>
                <option value="cpm">CPM (Cost per Mille)</option>
                <option value="cpa">CPA (Cost per Acquisition)</option>
                <option value="smart">Smart Bidding (ML)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Max Bid ($)</label>
              <input
                type="number"
                required
                min="0.01"
                step="0.01"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="1.50"
                value={formData.bidAmount}
                onChange={(e) => setFormData({...formData, bidAmount: e.target.value})}
              />
            </div>
          </div>
        </div>

        {/* Ad Format */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Ad Format</h3>
          <div className="grid grid-cols-4 gap-4">
            {['banner', 'video', 'native', 'interstitial'].map((format) => (
              <label key={format} className="relative flex items-center justify-center p-4 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  name="adFormat"
                  value={format}
                  className="sr-only"
                  checked={formData.adFormat === format}
                  onChange={(e) => setFormData({...formData, adFormat: e.target.value})}
                />
                <span className={`text-sm font-medium capitalize ${formData.adFormat === format ? 'text-primary-600' : 'text-gray-700'}`}>
                  {format}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Dates */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Schedule</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Start Date</label>
              <input
                type="date"
                required
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                value={formData.startDate}
                onChange={(e) => setFormData({...formData, startDate: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">End Date (Optional)</label>
              <input
                type="date"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                value={formData.endDate}
                onChange={(e) => setFormData({...formData, endDate: e.target.value})}
              />
            </div>
          </div>
        </div>

        {/* Targeting */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Targeting (Optional)</h3>
          <div className="grid grid-cols-1 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Countries</label>
              <input
                type="text"
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                placeholder="US, CA, GB (comma separated)"
                value={formData.targeting.countries.join(', ')}
                onChange={(e) => setFormData({
                  ...formData,
                  targeting: {
                    ...formData.targeting,
                    countries: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                  }
                })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Devices</label>
              <div className="mt-2 flex gap-4">
                {['mobile', 'desktop', 'tablet'].map((device) => (
                  <label key={device} className="inline-flex items-center">
                    <input
                      type="checkbox"
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                      checked={formData.targeting.devices.includes(device)}
                      onChange={(e) => {
                        const devices = e.target.checked
                          ? [...formData.targeting.devices, device]
                          : formData.targeting.devices.filter(d => d !== device)
                        setFormData({...formData, targeting: {...formData.targeting, devices}})
                      }}
                    />
                    <span className="ml-2 text-sm text-gray-700 capitalize">{device}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="flex justify-end pt-4 border-t border-gray-200">
          <button
            type="button"
            onClick={() => navigate('/campaigns')}
            className="mr-4 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
          >
            <Save className="w-4 h-4 mr-2" />
            Create Campaign
          </button>
        </div>
      </form>
    </div>
  )
}
